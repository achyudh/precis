import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.code2seq import NodeEncoder, ContextDecoder


class Code2Seq(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config

        self.node_embedding = nn.Embedding(vocab.node_vocab.size, config.node_embedding_dim)
        self.subtoken_embedding = nn.Embedding(vocab.subtoken_vocab.size, config.subtoken_embedding_dim)
        self.context_linear = nn.Linear(2 * (config.subtoken_embedding_dim + config.encoder_hidden_dim),
                                        config.decoder_hidden_dim, bias=False)

        self.encoder = NodeEncoder(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.decoder = ContextDecoder(config, vocab)

    def forward(self, source_subtoken_indices, node_indices, target_subtoken_indices, source_subtoken_lengths,
                node_lengths, target_subtoken_lengths, context_valid_mask):
        source_subtoken_embed = self.subtoken_embedding(source_subtoken_indices)  # (batch, max_contexts, max_subtokens, subtoken_embedding_dim)
        target_subtoken_embed = self.subtoken_embedding(target_subtoken_indices)  # (batch, max_contexts, max_subtokens, subtoken_embedding_dim)
        node_embed = self.node_embedding(node_indices)  # (batch, max_contexts, max_path_nodes, node_embedding_dim)

        source_subtoken_mask = torch.unsqueeze(self.sequence_mask(
            source_subtoken_lengths, maxlen=self.config.max_subtokens, dtype=torch.float32), -1)  # (batch, max_contexts, max_subtokens, 1)
        target_subtoken_mask = torch.unsqueeze(self.sequence_mask(
            target_subtoken_lengths, maxlen=self.config.max_subtokens, dtype=torch.float32), -1)  # (batch, max_contexts, max_subtokens, 1)

        source_subtoken_agg = torch.sum(source_subtoken_embed * source_subtoken_mask, dim=2)  # (batch, max_contexts, subtoken_embedding_dim)
        target_subtoken_agg = torch.sum(target_subtoken_embed * target_subtoken_mask, dim=2)  # (batch, max_contexts, subtoken_embedding_dim)
        node_agg = self.encoder(node_embed, node_lengths, context_valid_mask)  # (batch, max_contexts, max_path_nodes, encoder_hidden_dim)
        context_embed = torch.cat([source_subtoken_agg, node_agg, target_subtoken_agg], dim=-1)  # (batch, max_contexts, context_embed_dim)
        context_embed = self.dropout(context_embed)
        context_embed = F.tanh(self.context_linear(context_embed))  # (batch, max_contexts, decoder_hidden_dim)

        # Get initial decoder input and hidden state
        decoder_input = self.decoder.init_input()  # (batch,)
        h_0, c_0 = self.decoder.init_hidden(context_embed, context_valid_mask)  # (batch, decoder_hidden_dim)

        logits = list()
        for _ in range(self.config.max_target_length):
            decoder_output, h_0, c_0 = self.decoder(decoder_input, h_0, c_0, context_embed)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # Detach from history as input
            logits.append(decoder_output)

        return logits

    def sequence_mask(self, lengths, maxlen, dtype=torch.bool):
        cumsum = torch.ones((*lengths.shape, maxlen), device=self.config.device).cumsum(dim=1)
        mask = cumsum <= torch.unsqueeze(lengths, -1).type(torch.float32)
        return mask.type(dtype)
