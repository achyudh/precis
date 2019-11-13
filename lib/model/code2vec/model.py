import torch
import torch.nn as nn
import torch.nn.functional as F


class Code2Vec(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config

        self.path_embedding = nn.Embedding(vocab.path_vocab.size, config.path_embedding_dim)
        self.token_embedding = nn.Embedding(vocab.token_vocab.size, config.token_embedding_dim)
        self.input_dropout = nn.Dropout(config.dropout_rate)
        self.input_linear = nn.Linear(config.context_vector_size, config.context_vector_size, bias=False)
        self.attention = nn.Linear(config.context_vector_size, 1)
        self.output_linear = nn.Linear(config.context_vector_size, vocab.target_vocab.size)

    def forward(self, source_token_indices, path_indices, target_token_indices, context_valid_mask):
        source_token_embed = self.token_embedding(source_token_indices)  # (batch, max_contexts, token_embedding_dim)
        path_embed = self.path_embedding(path_indices)  # (batch, max_contexts, path_embedding_dim)
        target_token_embed = self.token_embedding(target_token_indices)  # (batch, max_contexts, token_embedding_dim)
        context_embed = torch.cat([source_token_embed, path_embed, target_token_embed], dim=-1)  # (batch, max_contexts, context_vector_size)
        context_embed = self.input_dropout(context_embed)

        context_embed = torch.reshape(context_embed, (-1, self.config.context_vector_size))  # (batch * max_contexts, context_vector_size)
        context_embed = F.tanh(self.input_linear(context_embed))  # (batch * max_contexts, context_vector_size)
        context_weights = self.attention(context_embed)  # (batch * max_contexts, 1)
        context_weights = torch.reshape(context_weights, (-1, self.config.max_contexts, 1))  # (batch, max_contexts, 1)
        context_weights += torch.unsqueeze(torch.log(context_valid_mask), dim=2)  # (batch, max_contexts, 1)
        attention_weights = F.softmax(context_weights, dim=1)  # (batch, max_contexts, 1)

        code_embed = torch.reshape(context_embed, (-1, self.config.max_contexts, self.config.context_vector_size))  # (batch, max_contexts, context_vector_size)
        code_embed = torch.sum(code_embed * attention_weights, dim=1)  # (batch, context_vector_size)
        logits = self.output_linear(code_embed)  # (batch, target_vocab_size)

        return logits
