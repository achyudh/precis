import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.data.vocab import PathContextVocabContainer


class ContextDecoder(nn.Module):
    def __init__(self, config, vocab: PathContextVocabContainer):
        super(ContextDecoder, self).__init__()
        self.config = config
        self.vocab = vocab

        self.target_embedding = nn.Embedding(vocab.target_vocab.size, config.target_embedding_dim)
        self.target_sos_index = self.vocab.target_vocab.word_to_index[self.vocab.target_vocab.special_words.SOS]

        self.attention_linear = nn.Linear(2 * config.decoder_hidden_dim, 1)
        self.attn_combine = nn.Linear(config.target_embedding_dim + config.decoder_hidden_dim, config.decoder_hidden_dim)
        self.lstm = nn.LSTMCell(config.decoder_hidden_dim, config.decoder_hidden_dim)
        self.output_linear = nn.Linear(config.decoder_hidden_dim, vocab.target_vocab.size)

    def forward(self, x, h_0, c_0, contexts):
        attn_input = torch.cat([contexts, h_0.unsqueeze(1).expand(h_0.shape[0], contexts.shape[1], h_0.shape[1])], dim=2)
        attn_weights = F.softmax(self.attention_linear(attn_input))  # (batch, max_contexts, 1)
        attn_applied = torch.sum(contexts * attn_weights, dim=1)  # (batch, decoder_hidden_dim)

        x = self.target_embedding(x)  # (batch, target_embedding_dim)
        x = torch.cat((x, attn_applied), dim=1)  # (batch, target_embedding_dim + decoder_hidden_dim)
        x = F.relu(self.attn_combine(x))  # (batch, decoder_hidden_dim)

        h_0, c_0 = self.lstm(x, (h_0, c_0))  # (batch, decoder_hidden_dim)
        output = self.output_linear(h_0)  # (batch, target_vocab_size)
        return output, h_0, c_0

    def init_input(self, batch_size):
        return torch.tensor([self.target_sos_index] * batch_size, device=self.config.device)  # (batch, 1)

    def init_state(self, contexts, context_valid_mask, batch_size):
        context_sum = torch.sum(contexts, dim=1)  # (batch, decoder_hidden_dim)
        context_counts = torch.sum(context_valid_mask, dim=1).unsqueeze(-1)  # (batch, 1)

        h_0 = context_sum / context_counts  # (batch, decoder_hidden_dim)
        c_0 = torch.zeros((batch_size, self.config.decoder_hidden_dim), device=self.config.device)
        return h_0, c_0
