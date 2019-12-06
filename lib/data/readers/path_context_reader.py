from argparse import Namespace
from typing import Tuple

import torch

from lib.data import PathContextInput
from lib.data.readers.context_reader import ContextReader
from lib.data.vocab import Code2VecVocabContainer


class PathContextReader(ContextReader):
    def __init__(self, config: Namespace, vocab: Code2VecVocabContainer):
        super().__init__(config)
        self.vocab = vocab

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

    def _get_input_tensors(self, index, *row_parts) -> Tuple:
        row_parts = list(row_parts)
        target_label = row_parts[0]
        target_index = torch.tensor(self.vocab.target_vocab.lookup_index(target_label))

        split_contexts = [x.split(',') for x in row_parts[1: self.config.max_contexts + 1]]

        token_pad_string = self.vocab.token_vocab.special_words.PAD
        token_pad_index = self.vocab.token_vocab.word_to_index[token_pad_string]
        path_pad_string = self.vocab.path_vocab.special_words.PAD
        path_pad_index = self.vocab.path_vocab.word_to_index[path_pad_string]

        source_token_indices = [self.vocab.token_vocab.lookup_index(x[0]) for x in split_contexts]
        source_token_indices += [token_pad_index for _ in range(self.config.max_contexts - len(source_token_indices))]
        source_token_indices = torch.tensor(source_token_indices)

        path_indices = [self.vocab.path_vocab.lookup_index(x[1]) for x in split_contexts]
        path_indices += [path_pad_index for _ in range(self.config.max_contexts - len(path_indices))]
        path_indices = torch.tensor(path_indices)

        target_token_indices = [self.vocab.token_vocab.lookup_index(x[2]) for x in split_contexts]
        target_token_indices += [token_pad_index for _ in range(self.config.max_contexts - len(target_token_indices))]
        target_token_indices = torch.tensor(target_token_indices)

        context_valid_mask = (torch.ne(source_token_indices, token_pad_index) |
                              torch.ne(target_token_indices, token_pad_index) |
                              torch.ne(path_indices, path_pad_index)).float()

        return PathContextInput(
            path_indices=path_indices,
            source_token_indices=source_token_indices,
            target_token_indices=target_token_indices,
            context_valid_mask=context_valid_mask,
            target_index=target_index,
            sample_index=torch.tensor(index)
        ), target_label