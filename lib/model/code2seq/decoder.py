import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextDecoder(nn.Module):
    def __init__(self, config, vocab):
        super(ContextDecoder, self).__init__()
        self.config = config
        self.vocab = vocab

        self.target_embedding = nn.Embedding(vocab.target_vocab.size, config.target_embedding_dim)
        self.attention = nn.Linear(config.target_embedding_dim + config.decoder_hidden_dim, config.max_contexts)
        self.attn_combine = nn.Linear(config.target_embedding_dim + config.decoder_hidden_dim, config.decoder_hidden_dim)
        self.lstm = nn.LSTMCell(config.decoder_hidden_dim, config.decoder_hidden_dim)
        self.output_linear = nn.Linear(config.decoder_hidden_dim, vocab.target_vocab.size)

    def forward(self, x, h_0, c_0, contexts):
        x = self.target_embedding(x)  # (batch, target_embedding_dim)
        attn_weights = F.softmax(self.attention(torch.cat((x, h_0), dim=1)), dim=1)  # (batch, max_contents)
        attn_applied = torch.sum(contexts * attn_weights.unsqueeze(2), dim=1)  # (batch, decoder_hidden_dim)

        x = torch.cat((x, attn_applied), dim=1)  # (batch, target_embedding_dim + decoder_hidden_dim)
        x = F.relu(self.attn_combine(x))  # (batch, decoder_hidden_dim)

        h_0, c_0 = self.lstm(x, (h_0, c_0))  # (batch, decoder_hidden_dim)
        output = self.output_linear(h_0)  # (batch, target_vocab_size)
        return output, h_0, c_0

    def init_input(self):
        target_sos_string = self.vocab.target_vocab.special_words.SOS
        target_sos_index = self.vocab.target_vocab.word_to_index[target_sos_string]
        return torch.tensor([target_sos_index] * self.config.batch_size, device=self.config.device)  # (batch,)

    def init_hidden(self, contexts, context_valid_mask):
        context_sum = torch.sum(contexts * torch.unsqueeze(context_valid_mask, -1), dim=1)  # (batch, decoder_hidden_dim)
        h_0 = context_sum / self.config.max_contexts
        c_0 = torch.zeros((self.config.batch_size, self.config.decoder_hidden_dim), device=self.config.device)
        return h_0, c_0
