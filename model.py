import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import to_var

class DialLV(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, word_dropout, pad_idx,
                sos_idx, eos_idx, max_utterance_length):

        super(DialLV, self).__init__()

        self.latent_size = latent_size

        self.encoder_embedding = nn.Embedding(vocab_size, embedding_size)

        self.prompt_encoder = Encoder(self.encoder_embedding, embedding_size, hidden_size)
        self.reply_encoder = Encoder(self.encoder_embedding, embedding_size, hidden_size)

        self.linear_mean = nn.Linear(hidden_size*2, latent_size)
        self.linear_log_var = nn.Linear(hidden_size*2, latent_size)

        # Reply Decoder
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
            word_dropout=word_dropout,
            pad_idx=pad_idx,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            max_utterance_length=max_utterance_length
            )


    def forward(self, prompt_sequece, prompt_length, reply_sequence, reply_length):

        batch_size = prompt_sequece.size(0)

        # Encode
        prompt_state = self.prompt_encoder(prompt_sequece, prompt_length)
        reply_state = self.reply_encoder(reply_sequence, reply_length)

        state = torch.cat((prompt_state, reply_state), dim=-1)

        # latent space parameters
        means = self.linear_mean(state)
        log_var = self.linear_log_var(state)
        std = torch.exp(0.5 * log_var)

        # Reparm.
        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + means

        # Decode
        out = self.decoder(reply_sequence, reply_length, prompt_state, z)

        return out, means, log_var

    def inference(self, prompt_sequece, prompt_length):

        prompt_state = self.prompt_encoder(prompt_sequece, prompt_length)

        batch_size = prompt_sequece.size(0)
        z = to_var(torch.randn([batch_size, self.latent_size]))

        out = self.decoder.inference(prompt_state, z)

        return out


class Encoder(nn.Module):

    def __init__(self, shared_embedding, embedding_size, hidden_size, bidirectional=True):

        super(Encoder, self).__init__()

        self.encoder_embedding = shared_embedding

        self.RNN = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size*2, hidden_size) # not clear from paper what the out dimensionaltiy is here

    def forward(self, input_sequence, input_length):

        batch_size = input_sequence.size(0)

        # sort inputs by length
        input_length, idx = input_length.sort(0, descending=True)
        input_sequence = input_sequence[idx]

        # embedd input sequence
        input_embedding = self.encoder_embedding(input_sequence)

        # RNN forward pass
        packed_inputs = pack_padded_sequence(input_embedding, input_length.data.tolist(), batch_first=True)
        _, last_encoder_hidden = self.RNN(packed_inputs, hx=None)

        # undo sorting
        _, reverse_idx = idx.sort()
        last_encoder_hidden = last_encoder_hidden[:,reverse_idx.data]

        # concat the states from bidirectioal
        last_encoder_hidden = torch.cat((last_encoder_hidden[0], last_encoder_hidden[1]), dim=-1)

        # transform and activate
        out = nn.functional.tanh(self.linear(last_encoder_hidden))

        return out

class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, word_dropout, pad_idx,
                sos_idx, eos_idx, max_utterance_length):

        super(Decoder, self).__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_utterance_length = max_utterance_length
        self.sample_mode = 'greedy'

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.word_dropout = nn.Dropout(p=word_dropout)

        self.RNN = nn.GRU(embedding_size, hidden_size + latent_size, batch_first=True)

        self.out = nn.Linear(hidden_size+latent_size, vocab_size)

    def forward(self, input_sequence, input_length, hx, z):

        # sort inputs by length
        input_length, idx = input_length.sort(0, descending=True)

        hx = hx[idx]
        z = z[idx]
        inital_hidden = torch.cat((hx, z), dim=-1)
        inital_hidden = inital_hidden.unsqueeze(0)

        input_sequence = input_sequence[idx]

        # embedd input sequence
        input_embedding = self.embedding(input_sequence)
        input_embedding = self.word_dropout(input_embedding)
        # RNN forwardpass
        packed_inputs = pack_padded_sequence(input_embedding, input_length.data.tolist(), batch_first=True)
        outputs, _ = self.RNN(packed_inputs, hx=inital_hidden)

        logits = self.out(outputs.data)

        log_probs = torch.nn.functional.log_softmax(logits)

        return log_probs

    def inference(self, hx, z):
        """Inference mode of Decoder, no gold reply is provided, therefore sample token at t-1 will
        be input to current timestep.

        Parameters
        ----------
        hx : Variable(torch.FloatTensor)
            Hidden state of Prompt encoder.
        z : Variable(torch.FloatTensor)
            Sampled latent variable from standard Gaussian.

        Returns
        -------
        Variable(torch.LongTensor)
            Generated sequence.

        """

        hidden = torch.cat((hx, z), dim=-1)
        hidden = hidden.unsqueeze(0)

        batch_size = hx.size(0)

        # required for dynamic stopping of reply generation
        sequence_idx = torch.arange(0, batch_size).long().cuda() if torch.cuda.is_available() else torch.arange(0, batch_size).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size).long().cuda() if torch.cuda.is_available() else torch.arange(0, batch_size).long()# all idx of batch wich are still generating
        sequence_mask = torch.ones(batch_size).byte().cuda() if torch.cuda.is_available() else torch.ones(batch_size).byte()

        running_seqs = torch.arange(0, batch_size).long().cuda() if torch.cuda.is_available() else torch.arange(0, batch_size).long() # idx of still generating sequences with respect to current loop
        #running_mask = torch.ones(batch_size).byte()

        replies = torch.Tensor(batch_size, self.max_utterance_length).fill_(self.pad_idx).long().cuda() if torch.cuda.is_available() else torch.Tensor(batch_size, self.max_utterance_length).fill_(self.pad_idx).long()

        t = 0
        while(len(running_seqs) > 0 and t<self.max_utterance_length):

            if t == 0:
                input = to_var(torch.Tensor([self.sos_idx] * batch_size).long())

            input_embedding = self.embedding(input.unsqueeze(1))

            if t > 0:
                hidden = hidden.transpose(1,0)

            outputs, hidden = self.RNN(input_embedding, hidden)
            hidden = hidden.transpose(1,0)

            logits = self.out(outputs)
            log_probs = torch.nn.functional.log_softmax(logits)

            # get next input
            input = self._sample(log_probs)

            # save next input
            replies = self._save_sample(replies, input, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input = input[running_seqs]
                hidden = hidden[running_seqs]

                running_seqs = torch.arange(0, len(running_seqs)).long().cuda() if torch.cuda.is_available() else torch.arange(0, len(running_seqs)).long()

            t += 1

        return replies

    def _sample(self, predictions):
        """Samples from predictions distribution.

        Parameters
        ----------
        predictions : torch.Tensor or Variable(torch.Tensor)
            Two dimenionsal tensor where last dimension is distribution.

        Returns
        -------
        torch.LongTensor or Variable(torch.LongTensor)
            One dimensional tensor with idx according to sample

        """

        if self.sample_mode == 'greedy':
            _, sample = torch.topk(predictions, 1, dim=-1)
            sample = sample.squeeze()

        else:
            raise NotImplementedError("Sample method %s not implemented."%self.sample_mode)

            # TODO add sampling from distribution


        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        """Saves a sample into a `save_to` at current timestep (t), given the sequences which are
        still generating (running).

        Parameters
        ----------
        save_to : torch.LongTensor
            Tensor of size [batch x sequence]; holds all previous samples.
        sample : torch.LongTensor
            Tensor of size [batch]; holds samples from current timestep.
        running : torch.LongTensor
            Tensor containing the idicies of still running sequences.
        t : int
            Current timestep.

        Returns
        -------
        torch.LongTensor
            Updated `save_to` Tensor, with sample inserted at current timestep.

        """
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
