import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import to_var

class DialLV(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size):

        super(DialLV, self).__init__()

        self.latent_size = latent_size

        self.encoder_embedding = nn.Embedding(vocab_size, embedding_size)

        self.prompt_encoder = Encoder(self.encoder_embedding, embedding_size, hidden_size)
        self.reply_encoder = Encoder(self.encoder_embedding, embedding_size, hidden_size)

        self.linear_mean = nn.Linear(hidden_size*4, latent_size)
        self.linear_log_var = nn.Linear(hidden_size*4, latent_size)

        # Reply Decoder
        self.decoder= Decoder(vocab_size, embedding_size, hidden_size, latent_size)

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


class Encoder(nn.Module):

    def __init__(self, shared_embedding, embedding_size, hidden_size, bidirectional=True):

        super(Encoder, self).__init__()

        self.encoder_embedding = shared_embedding

        self.RNN = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size*2, hidden_size*2) # not clear from paper what the out dimensionaltiy is here

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

    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size):

        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.RNN = nn.GRU(embedding_size, hidden_size*2 + latent_size, batch_first=True)

        self.out = nn.Linear(hidden_size*2+latent_size, vocab_size)

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

        # RNN forwardpass
        packed_inputs = pack_padded_sequence(input_embedding, input_length.data.tolist(), batch_first=True)
        outputs, _ = self.RNN(packed_inputs, hx=inital_hidden)
        #outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        logits = self.out(outputs.data)

        log_probs = torch.nn.functional.log_softmax(logits)

        return log_probs
