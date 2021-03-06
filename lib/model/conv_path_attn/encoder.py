import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(config.node_embedding_dim, config.encoder_hidden_dim, dropout=config.dropout_rate,
                            bidirectional=True, batch_first=True)

    def forward(self, node_embed, node_lengths):
        node_lengths = torch.reshape(node_lengths, (-1,))  # (batch * max_contexts)

        # (batch * max_contexts, max_path_nodes, node_embedding_dim)
        x = torch.reshape(node_embed, shape=(-1, self.config.max_path_nodes, self.config.node_embedding_dim))

        x = torch.nn.utils.rnn.pack_padded_sequence(x, node_lengths, batch_first=True, enforce_sorted=False)
        x, hidden = self.lstm(x)

        # (batch, max_contexts, encoder_hidden_dim)
        return torch.reshape(hidden[0], shape=(-1, self.config.max_contexts, 2 * self.config.encoder_hidden_dim))
