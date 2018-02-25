import time
import torch
import argparse
import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from model import DialLV
from utils import to_var, idx2word, save_dial_to_json
from OpenSubtitlesQADataset import OpenSubtitlesQADataset

def main(args):

    splits = ['train', 'valid']

    datasets = dict()
    for split in splits:
        datasets[split] = OpenSubtitlesQADataset(
            root='data',
            split=split,
            min_occ=args.min_occ,
            max_utterance_length=args.max_utterance_length
            )

    model = DialLV(vocab_size=datasets['train'].vocab_size,
                    embedding_size=args.embedding_size,
                    hidden_size=args.hidden_size,
                    latent_size=args.latent_size,
                    word_dropout=args.word_dropout,
                    pad_idx=datasets['train'].pad_idx,
                    sos_idx=datasets['train'].sos_idx,
                    eos_idx=datasets['train'].eos_idx,
                    max_utterance_length=args.max_utterance_length
                    )

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

        random_question_idx = np.random.choice(np.arange(0, len(train_dataset)), 10, replace=False).astype('int64')
        random_questions = np.zeros((n, args.max_utterance_length)).astype('int64')
        random_questions_length = np.zeros(n)
        for i, rqi in enumerate(random_question_idx):
            random_questions[i] = train_dataset[rqi]['question']
            random_questions_length[i] = train_dataset[rqi]['question_length']

        input_sequence = to_var(torch.from_numpy(random_questions))
        input_length = to_var(torch.from_numpy(random_questions_length))
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

            data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=split=='train')

            tracker = defaultdict(torch.Tensor)

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

                question = batch['question']
                question_length = batch['question_length']
                answer_input = batch['answer_input']
                answer_target = batch['answer_target']
                answer_length = batch['answer_length']

                # model forward pass
                predictions, mean, log_var = model(
                    prompt_sequece=question,
                    prompt_length=question_length,
                    reply_sequence=answer_input,
                    reply_length=answer_length
                    )

                # predictions come back packed, so making targets packed as well to ignore all padding tokens
                sorted_length, sort_idx = answer_length.sort(0, descending=True)
                targets = answer_target[sort_idx]
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
                tracker['loss'] = torch.cat((tracker['loss'], loss.data))
                tracker['nll_loss'] = torch.cat((tracker['nll_loss'], nll_loss.data))
                tracker['kl_weighted_loss'] = torch.cat((tracker['kl_weighted_loss'], kl_weighted_loss.data))
                tracker['kl_weight'] = torch.cat((tracker['kl_weight'], torch.Tensor([kl_weight])))
                tracker['kl_loss'] = torch.cat((tracker['kl_loss'], kl_loss.data))

                if args.tensorboard_logging:
                    writer.add_scalar("%s/Batch-Loss"%(split), loss.data[0], epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/Batch-NLL-Loss"%(split), nll_loss.data[0], epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/Batch-KL-Loss"%(split), kl_loss.data[0], epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/Batch-KL-Weight"%(split), kl_weight, epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/Batch-KL-Loss-Weighted"%(split), kl_weighted_loss.data[0], epoch * len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL Loss %9.4f, KL Loss %9.4f, KLW Loss %9.4f, w %6.4f"
                        %(split.upper(), iteration, len(data_loader), loss.data[0], nll_loss.data[0], kl_loss.data[0], kl_weighted_loss.data[0], kl_weight))

                    prompts, replies = inference(model, datasets['train'])
                    save_dial_to_json(prompts, replies, root="dials/"+str(ts)+"/", comment="E"+str(epoch) + "I"+str(iteration))

            print("%s Epoch %02d/%i, Mean Loss: %.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['loss'])))

            # TODO: save model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--create_data", action='store_true')
    parser.add_argument("--min_occ", type=int, default=50)
    parser.add_argument("--max_utterance_length", type=int, default=30)
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
