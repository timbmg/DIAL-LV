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

def inference(model, train_dataset, n=10):
    random_question_idx = np.random.choice(np.arange(0, len(train_dataset)), 10, replace=False).astype('int64')
    random_questions = np.zeros((n, args.max_utterance_length)).astype('int64')
    random_questions_length = np.zeros(n)
    for i, rqi in enumerate(random_question_idx):
        random_questions[i] = train_dataset[i]['question']
        random_questions_length[i] = train_dataset[i]['question_length']

    input_sequence = to_var(torch.from_numpy(random_questions))
    input_length = to_var(torch.from_numpy(random_questions_length))
    prompts = idx2word(input_sequence.data, train_dataset.i2w)

    replies = model.inference(input_sequence, input_length)
    replies = idx2word(replies, train_dataset.i2w)

    return prompts, replies

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
                    embedding_size=300,
                    hidden_size=256,
                    latent_size=64,
                    pad_idx=datasets['train'].pad_idx,
                    sos_idx=datasets['train'].sos_idx,
                    eos_idx=datasets['train'].eos_idx,
                    max_utterance_length=args.max_utterance_length
                    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, size_average=False)

    def kl_anneal_function(k, x0, x):
        # https://en.wikipedia.org/wiki/Logistic_function
        return float(1/(1+np.exp(-k*(x-x0))))

    def loss_fn(predictions, targets, mean, log_var, k, x0, x):

        nll_loss = NLL(predictions, targets)

        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        kl_weight = kl_anneal_function(k, x0, x)

        kl_weighted = kl_weight * kl_loss

        return nll_loss, kl_weighted, kl_weight, kl_loss

    if args.tensorboard_logging:
        writer = SummaryWriter("logs/"+str(time.time()))

    global_step = 0
    for epoch in range(args.epochs):

        for split, dataset in datasets.items():

            data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=split=='train')

            tracker = defaultdict(torch.Tensor)

            for iteration, batch in enumerate(data_loader):

                # get batch items
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                question = batch['question']
                question_length = batch['question_length']
                answer = batch['answer']
                answer_length = batch['answer_length']


                predictions, mean, log_var = model(
                    prompt_sequece=question,
                    prompt_length=question_length,
                    reply_sequence=answer,
                    reply_length=answer_length
                    )

                sorted_length, sort_idx = answer_length.sort(0, descending=True)
                targets = answer[sort_idx]
                targets = pack_padded_sequence(targets, sorted_length.data.tolist(), batch_first=True)[0]

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
                    print("%s Batch %04d/%i, Loss %.4f, NLL Loss %.4f, KL Loss %.4f, KL_w Loss %.4f, w %.4f"
                        %(split.upper(), iteration, len(data_loader), loss.data[0], nll_loss.data[0], kl_loss.data[0], kl_weighted_loss.data[0], kl_weight))

                    prompts, replies = inference(model, datasets['train'])
                    save_dial_to_json(prompts, replies, root="dials", comment="E"+str(epoch) + "I"+str(iteration))

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
    parser.add_argument("--kl_anneal_k", type=float, default=0.00025, help="Steepness of Annealing function")
    parser.add_argument("--kl_anneal_x0", type=int, default=15000, help="Midpoint of Annealing function (i.e. weight=0.5)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--tensorboard_logging", action='store_true')


    args = parser.parse_args()

    main(args)
