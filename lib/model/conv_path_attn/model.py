import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.conv_path_attn import NodeEncoder


class ConvPathAttn(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.encoder = NodeEncoder(config)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.attention = nn.Linear(config.decoder_hidden_dim, 1)
        self.output_linear = nn.Linear(config.decoder_hidden_dim, vocab.target_vocab.size)
        self.node_embedding = nn.Embedding(vocab.node_vocab.size, config.node_embedding_dim)
        self.subtoken_embedding = nn.Embedding(vocab.subtoken_vocab.size, config.subtoken_embedding_dim)
        self.context_linear = nn.Linear(2 * (config.subtoken_embedding_dim + config.encoder_hidden_dim),
                                        config.decoder_hidden_dim, bias=False)

    def forward(self, source_subtoken_indices, node_indices, target_subtoken_indices, source_subtoken_lengths,
                node_lengths, target_subtoken_lengths, context_valid_mask):
        source_subtoken_embed = self.subtoken_embedding(source_subtoken_indices)  # (batch, max_contexts, max_subtokens, subtoken_embedding_dim)
        target_subtoken_embed = self.subtoken_embedding(target_subtoken_indices)  # (batch, max_contexts, max_subtokens, subtoken_embedding_dim)
        node_embed = self.node_embedding(node_indices)  # (batch, max_contexts, max_path_nodes, node_embedding_dim)

        source_subtoken_mask = self.sequence_mask(source_subtoken_lengths, max_len=self.config.max_subtokens)  # (batch, max_contexts, max_subtokens, 1)
        target_subtoken_mask = self.sequence_mask(target_subtoken_lengths, max_len=self.config.max_subtokens)  # (batch, max_contexts, max_subtokens, 1)

        source_subtoken_agg = torch.sum(source_subtoken_embed * source_subtoken_mask, dim=2)  # (batch, max_contexts, subtoken_embedding_dim)
        target_subtoken_agg = torch.sum(target_subtoken_embed * target_subtoken_mask, dim=2)  # (batch, max_contexts, subtoken_embedding_dim)
        node_agg = self.encoder(node_embed, node_lengths) * context_valid_mask.unsqueeze(-1)  # (batch, max_contexts, max_path_nodes, encoder_hidden_dim)

        context_embed = torch.cat([source_subtoken_agg, node_agg, target_subtoken_agg], dim=-1)  # (batch, max_contexts, context_embed_dim)
        context_embed = self.dropout(context_embed)
        context_embed = F.tanh(self.context_linear(context_embed))  # (batch, max_contexts, decoder_hidden_dim)

        context_weights = self.attention(context_embed)  # (batch * max_contexts, 1)
        context_weights = torch.reshape(context_weights, (-1, self.config.max_contexts, 1))  # (batch, max_contexts, 1)
        context_weights += torch.unsqueeze(torch.log(context_valid_mask), dim=2)  # (batch, max_contexts, 1)
        attention_weights = F.softmax(context_weights, dim=1)  # (batch, max_contexts, 1)

        code_embed = torch.reshape(context_embed, (-1, self.config.max_contexts, self.config.decoder_hidden_dim))  # (batch, max_contexts, decoder_hidden_dim)
        code_embed = torch.sum(code_embed * attention_weights, dim=1)  # (batch, decoder_hidden_dim)
        logits = self.output_linear(code_embed)  # (batch, target_vocab_size)

        return logits

    def sequence_mask(self, lengths, max_len, dtype=torch.float32):
        mask = torch.arange(max_len, device=self.config.device).expand(*lengths.shape, max_len) < lengths.unsqueeze(-1)
        return mask.unsqueeze(-1).type(dtype)
