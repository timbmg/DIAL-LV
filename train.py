import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from torch.nn.utils.rnn import pack_padded_sequence

from model import DialLV
from utils import to_var, idx2word, save_dial_to_json
from OpenSubtitlesQADataset import OpenSubtitlesQADataset
from GuessWhatDataset import GuessWhatDataset

def main(args):

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    splits = ['train', 'valid']

    datasets = OrderedDict()
    for split in splits:
        if args.dataset.lower() == 'opensubtitles':
            datasets[split] = OpenSubtitlesQADataset(
                root='data',
                split=split,
                min_occ=args.min_occ,
                max_prompt_length=args.max_input_length,
                max_reply_length=args.max_reply_length
                )
        elif args.dataset.lower() == 'guesswhat':
            datasets[split] = GuessWhatDataset(
                root='data',
                split=split,
                min_occ=args.min_occ,
                max_dialogue_length=args.max_input_length,
                max_question_length=args.max_reply_length
                )

    model = DialLV(vocab_size=datasets['train'].vocab_size,
                    embedding_size=args.embedding_size,
                    hidden_size=args.hidden_size,
                    latent_size=args.latent_size,
                    word_dropout=args.word_dropout,
                    pad_idx=datasets['train'].pad_idx,
                    sos_idx=datasets['train'].sos_idx,
                    eos_idx=datasets['train'].eos_idx,
                    max_utterance_length=args.max_reply_length
                    )

    if torch.cuda.is_available():
        model = model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    NLL = torch.nn.NLLLoss(size_average=False)

    def kl_anneal_function(k, x0, x):
        """ Returns the weight of for calcualting the weighted KL Divergence.
        https://en.wikipedia.org/wiki/Logistic_function
        """
        return float(1/(1+np.exp(-k*(x-x0))))

    def loss_fn(predictions, targets, mean, log_var, k, x0, x):
        """Calcultes the ELBO, consiting of the Negative Log Likelihood and KL Divergence.

        Parameters
        ----------
        predictions : Variable(torch.FloatTensor) [? x vocab_size]
            Log probabilites of each generated token in the batch. Number of tokens depends on
            tokens in batch.
        targets : Variable(torch.LongTensor) [?]
            Target token ids. Number of tokens depends on tokens in batch.
        mean : Variable(torch.FloatTensor) [batch_size x latent_size]
            Predicted mean values of latent variables.
        log_var : Variable(torch.FloatTensor) [batch_size x latent_size]
            Predicted log variabnce values of latent variables.
        k : type
            Steepness parameter for kl weight calculation.
        x0 : type
            Midpoint parameter for kl weight calculation.
        x : int
            Global step.

        Returns
        -------
        Variable(torch.FloatTensor), Variable(torch.FloatTensor), float, Variable(torch.FloatTensor)
            NLLLoss value, weighted KL Divergence loss, weight value and unweighted KL Divergence.

        """

        nll_loss = NLL(predictions, targets)

        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        kl_weight = kl_anneal_function(k, x0, x)

        kl_weighted = kl_weight * kl_loss

        return nll_loss, kl_weighted, kl_weight, kl_loss

    def inference(model, train_dataset, n=10, m=3):
        """ Executes the model in inference mode and returns string of inputs and corresponding
        generations.

        Parameters
        ----------
        model : DIAL-LV
            The DIAL-LV model.
        train_dataset : type
            Training dataset to draw random input samples from.
        n : int
            Number of samples to draw.
        m : int
            Number of response generations.

        Returns
        -------
        string, string
            Two string, each consiting of n utterances. `Prompts` contains the input sequence and
            `replies` the generated response sequence.

        """

        random_input_idx = np.random.choice(np.arange(0, len(train_dataset)), 10, replace=False).astype('int64')
        random_inputs = np.zeros((n, args.max_input_length)).astype('int64')
        random_inputs_length = np.zeros(n)
        for i, rqi in enumerate(random_input_idx):
            random_inputs[i] = train_dataset[rqi]['input_sequence']
            random_inputs_length[i] = train_dataset[rqi]['input_length']

        input_sequence = to_var(torch.from_numpy(random_inputs).long())
        input_length = to_var(torch.from_numpy(random_inputs_length).long())
        prompts = idx2word(input_sequence.data, train_dataset.i2w, train_dataset.pad_idx)

        replies = list()
        for i in range(m):
            replies_ = model.inference(input_sequence, input_length)
            replies.append(idx2word(replies_, train_dataset.i2w, train_dataset.pad_idx))

        return prompts, replies

    ts = time.time()
    if args.tensorboard_logging:
        writer = SummaryWriter("logs/"+str(ts))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))

    global_step = 0
    for epoch in range(args.epochs):

        for split, dataset in datasets.items():

            data_loader = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
                )

            tracker = defaultdict(tensor)

            if split == 'train':
                model.train()
            else:
                # disable drop out when in validation
                model.eval()

            for iteration, batch in enumerate(data_loader):

                # get batch items and wrap them in variables
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                input_sequence = batch['input_sequence']
                input_length = batch['input_length']
                reply_sequence_in = batch['reply_sequence_in']
                reply_sequence_out = batch['reply_sequence_out']
                reply_length = batch['reply_length']
                batch_size = input_sequence.size(0)


                # model forward pass
                predictions, mean, log_var = model(
                    prompt_sequece=input_sequence,
                    prompt_length=input_length,
                    reply_sequence=reply_sequence_in,
                    reply_length=reply_length
                    )

                # predictions come back packed, so making targets packed as well to ignore all padding tokens
                sorted_length, sort_idx = reply_length.sort(0, descending=True)
                targets = reply_sequence_out[sort_idx]
                targets = pack_padded_sequence(targets, sorted_length.data.tolist(), batch_first=True)[0]

                # compute the loss
                nll_loss, kl_weighted_loss, kl_weight, kl_loss = loss_fn(predictions, targets, mean, log_var, args.kl_anneal_k, args.kl_anneal_x0, global_step)
                loss = nll_loss + kl_weighted_loss

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    global_step += 1

                # bookkeeping
                tracker['loss']             = torch.cat((tracker['loss'],               loss.data/batch_size))
                tracker['nll_loss']         = torch.cat((tracker['nll_loss'],           nll_loss.data/batch_size))
                tracker['kl_loss']          = torch.cat((tracker['kl_loss'],            kl_loss.data/batch_size))
                tracker['kl_weight']        = torch.cat((tracker['kl_weight'],          tensor([kl_weight])))
                tracker['kl_weighted_loss'] = torch.cat((tracker['kl_weighted_loss'],   kl_weighted_loss.data/batch_size))

                if args.tensorboard_logging:
                    step = epoch * len(data_loader) + iteration
                    writer.add_scalar("%s/Batch-Loss"%(split),              tracker['loss'][-1],                step)
                    writer.add_scalar("%s/Batch-NLL-Loss"%(split),          tracker['nll_loss'][-1],            step)
                    writer.add_scalar("%s/Batch-KL-Loss"%(split),           tracker['kl_loss'][-1],             step)
                    writer.add_scalar("%s/Batch-KL-Weight"%(split),         tracker['kl_weight'][-1],           step)
                    writer.add_scalar("%s/Batch-KL-Loss-Weighted"%(split),  tracker['kl_weighted_loss'][-1],    step)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL Loss %9.4f, KL Loss %9.4f, KLW Loss %9.4f, w %6.4f"
                        %(split.upper(), iteration, len(data_loader),
                        tracker['loss'][-1], tracker['nll_loss'][-1], tracker['kl_loss'][-1],
                        tracker['kl_weighted_loss'][-1], tracker['kl_weight'][-1]))

                    prompts, replies = inference(model, datasets['train'])
                    save_dial_to_json(prompts, replies, root="dials/"+str(ts)+"/", comment="E"+str(epoch) + "I"+str(iteration))

            print("%s Epoch %02d/%i, Mean Loss: %.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['loss'])))
            if args.tensorboard_logging:
                writer.add_scalar("%s/Epoch-Loss"%(split),      torch.mean(tracker['loss']),        epoch)
                writer.add_scalar("%s/Epoch-NLL-Loss"%(split),  torch.mean(tracker['nll_loss']),    epoch)
                writer.add_scalar("%s/Epoch-KL-Loss"%(split),   torch.mean(tracker['kl_loss']),     epoch)

            # TODO: save model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="OpenSubtitles")
    parser.add_argument("--create_data", action='store_true')
    parser.add_argument("--min_occ", type=int, default=3)
    parser.add_argument("--max_input_length", type=int, default=100)
    parser.add_argument("--max_reply_length", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--word_dropout", type=float, default=0.5)
    parser.add_argument("--kl_anneal_k", type=float, default=0.00025, help="Steepness of Annealing function")
    parser.add_argument("--kl_anneal_x0", type=int, default=15000, help="Midpoint of Annealing function (i.e. weight=0.5)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--tensorboard_logging", action='store_true')

    args = parser.parse_args()

    main(args)
