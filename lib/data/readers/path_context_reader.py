import random
from argparse import Namespace
from typing import Tuple

import torch

from lib.data import PathContextInput
from lib.data.readers.dataset_reader import DatasetReader
from lib.data.vocab.path_context_vocab_container import PathContextVocabContainer


class PathContextReader(DatasetReader):
    def __init__(self, config: Namespace, vocab: PathContextVocabContainer):
        super().__init__(config, vocab)
        self.path_pad_index = self.vocab.path_vocab.word_to_index[self.vocab.path_vocab.special_words.PAD]
        self.token_pad_index = self.vocab.token_vocab.word_to_index[self.vocab.token_vocab.special_words.PAD]

    def is_valid_input_row(self, input_tensor, split) -> bool:
        any_context_is_valid = (torch.max(input_tensor.source_token_indices).item() != self.token_pad_index |
                                torch.max(input_tensor.target_token_indices).item() != self.token_pad_index |
                                torch.max(input_tensor.path_indices).item() != self.path_pad_index)

        if split.is_dev or split.is_test:
            return any_context_is_valid
        else:
            target_oov_index = self.vocab.target_vocab.word_to_index[self.vocab.target_vocab.special_words.OOV]
            word_is_valid = input_tensor.label_index > target_oov_index
            return word_is_valid and any_context_is_valid

    def _get_input_tensors(self, index, *row_parts) -> Tuple:
        row_parts = list(row_parts)
        label_string = row_parts[0]
        label_index = torch.tensor(self.vocab.target_vocab.lookup_index(label_string))

        split_contexts = [x.split(',') for x in row_parts[1:]]

        if self.config.context_sampling:
            split_contexts = random.choices(split_contexts, k=self.config.max_contexts)
        else:
            split_contexts = split_contexts[:self.config.max_contexts]

        source_token_indices = [self.vocab.token_vocab.lookup_index(x[0]) for x in split_contexts]
        source_token_indices += [self.token_pad_index for _ in range(self.config.max_contexts - len(source_token_indices))]
        source_token_indices = torch.tensor(source_token_indices)

        path_indices = [self.vocab.path_vocab.lookup_index(x[1]) for x in split_contexts]
        path_indices += [self.path_pad_index for _ in range(self.config.max_contexts - len(path_indices))]
        path_indices = torch.tensor(path_indices)

        target_token_indices = [self.vocab.token_vocab.lookup_index(x[2]) for x in split_contexts]
        target_token_indices += [self.token_pad_index for _ in range(self.config.max_contexts - len(target_token_indices))]
        target_token_indices = torch.tensor(target_token_indices)

        context_valid_mask = (torch.ne(source_token_indices, self.token_pad_index) |
                              torch.ne(target_token_indices, self.token_pad_index) |
                              torch.ne(path_indices, self.path_pad_index)).float()

        return PathContextInput(
            label_index=label_index,
            path_indices=path_indices,
            source_token_indices=source_token_indices,
            target_token_indices=target_token_indices,
            context_valid_mask=context_valid_mask,
            sample_index=torch.tensor(index)
        ), label_string
