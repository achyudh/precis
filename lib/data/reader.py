from argparse import Namespace
from typing import NamedTuple, Optional

import torch

from lib.data.vocab import Code2VecVocab


class ReaderInputTensors(NamedTuple):
    """
    Used mostly for convenient-and-clear access to input parts (by their names).
    """
    source_token_indices: torch.Tensor
    path_indices: torch.Tensor
    target_token_indices: torch.Tensor
    context_valid_mask: torch.Tensor
    target_index: Optional[torch.Tensor] = None


class PathContextReader:
    def __init__(self, config: Namespace, vocab: Code2VecVocab):
        self.vocab = vocab
        self.config = config

    def process_input_row(self, row_placeholder) -> ReaderInputTensors:
        parts = row_placeholder.split()
        return self._map_raw_dataset_row_to_input_tensors(*parts)

    def is_valid_input_row(self, input_tensor, split) -> bool:
        token_pad_index = self.vocab.token_vocab.word_to_index[self.vocab.token_vocab.special_words.PAD]
        path_pad_index = self.vocab.path_vocab.word_to_index[self.vocab.path_vocab.special_words.PAD]

        any_context_is_valid = (torch.max(input_tensor.source_token_indices).item() != token_pad_index |
                                torch.max(input_tensor.target_token_indices).item() != token_pad_index |
                                torch.max(input_tensor.path_indices).item() != path_pad_index)

        if split.is_dev or split.is_test:
            return any_context_is_valid
        else:
            target_oov_index = self.vocab.target_vocab.word_to_index[self.vocab.target_vocab.special_words.OOV]
            word_is_valid = input_tensor.target_index > target_oov_index
            return word_is_valid and any_context_is_valid

    def _map_raw_dataset_row_to_input_tensors(self, *row_parts) -> ReaderInputTensors:
        row_parts = list(row_parts)
        target_str = row_parts[0]
        target_index = torch.tensor(self.vocab.target_vocab.lookup_index(target_str))

        split_contexts = [x.split(',') for x in row_parts[1: self.config.max_contexts + 1]]

        token_pad_string = self.vocab.token_vocab.special_words.PAD
        path_pad_string = self.vocab.path_vocab.special_words.PAD
        token_pad_index = self.vocab.token_vocab.word_to_index[token_pad_string]
        path_pad_index = self.vocab.path_vocab.word_to_index[path_pad_string]

        source_token_indices = [self.vocab.token_vocab.lookup_index(x[0]) for x in split_contexts]
        source_token_indices += [token_pad_index for _ in range(self.config.max_contexts - len(source_token_indices))]
        source_token_indices = torch.tensor(source_token_indices)

        path_indices = [self.vocab.token_vocab.lookup_index(x[1]) for x in split_contexts]
        path_indices += [path_pad_index for _ in range(self.config.max_contexts - len(path_indices))]
        path_indices = torch.tensor(path_indices)

        target_token_indices = [self.vocab.token_vocab.lookup_index(x[2]) for x in split_contexts]
        target_token_indices += [token_pad_index for _ in range(self.config.max_contexts - len(target_token_indices))]
        target_token_indices = torch.tensor(target_token_indices)

        context_valid_mask = (torch.ne(source_token_indices, token_pad_index) |
                              torch.ne(target_token_indices, token_pad_index) |
                              torch.ne(path_indices, path_pad_index)).float()

        return ReaderInputTensors(
            source_token_indices=source_token_indices,
            path_indices=path_indices,
            target_token_indices=target_token_indices,
            context_valid_mask=context_valid_mask,
            target_index=target_index,
        )
