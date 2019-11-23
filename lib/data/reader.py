import abc
from argparse import Namespace
from typing import NamedTuple, Tuple

import numpy as np
import torch

from lib.data.vocab import Code2VecVocabContainer, Code2SeqVocabContainer


class PathContextInput(NamedTuple):
    """
    Used mostly for convenient-and-clear access to input parts (by their names).
    """
    path_indices: torch.Tensor
    source_token_indices: torch.Tensor
    target_token_indices: torch.Tensor
    context_valid_mask: torch.Tensor
    target_index: torch.Tensor
    sample_index: torch.Tensor


class SequentialPathContextInput(NamedTuple):
    """
    Used mostly for convenient-and-clear access to input parts (by their names).
    """
    node_indices: torch.Tensor
    node_lengths: torch.Tensor
    source_subtoken_indices: torch.Tensor
    source_subtoken_lengths: torch.Tensor
    target_subtoken_indices: torch.Tensor
    target_subtoken_lengths: torch.Tensor
    context_valid_mask: torch.Tensor
    target_indices: torch.Tensor
    sample_index: torch.Tensor


class AbstractContextReader(abc.ABC):
    def __init__(self, config: Namespace):
        self.config = config

    def process_input_row(self, index, row_placeholder) -> Tuple:
        parts = row_placeholder.split()
        return self._map_raw_dataset_row_to_input_tensors(index, *parts)

    @abc.abstractmethod
    def _map_raw_dataset_row_to_input_tensors(self, *row_parts) -> Tuple:
        pass


class PathContextReader(AbstractContextReader):
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

    def _map_raw_dataset_row_to_input_tensors(self, index, *row_parts) -> Tuple:
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


class SequentialPathContextReader(AbstractContextReader):
    def __init__(self, config: Namespace, vocab: Code2SeqVocabContainer):
        super().__init__(config)
        self.vocab = vocab

    def _map_raw_dataset_row_to_input_tensors(self, index, *row_parts) -> Tuple:
        row_parts = list(row_parts)
        subtoken_pad_string = self.vocab.subtoken_vocab.special_words.PAD
        subtoken_pad_index = self.vocab.subtoken_vocab.word_to_index[subtoken_pad_string]
        node_pad_string = self.vocab.node_vocab.special_words.PAD
        node_pad_index = self.vocab.node_vocab.word_to_index[node_pad_string]
        target_pad_string = self.vocab.target_vocab.special_words.PAD
        target_pad_index = self.vocab.target_vocab.word_to_index[target_pad_string]

        target_label = row_parts[0]
        target_strings = target_label.split('|')[:self.config.max_target_length]
        target_strings = [self.vocab.target_vocab.lookup_index(x) for x in target_strings]
        target_strings += [target_pad_index for _ in range(self.config.max_target_length - len(target_strings))]
        target_indices = torch.tensor(target_strings)

        split_contexts = [x.split(',') for x in row_parts[1: self.config.max_contexts + 1]]

        source_subtoken_strings = [x[0].split('|') for x in split_contexts][:self.config.max_subtokens]
        source_subtoken_lengths = [len(string) for string in source_subtoken_strings]
        source_subtoken_lengths += [0 for _ in range(self.config.max_contexts - len(source_subtoken_strings))]
        source_subtoken_lengths = torch.tensor(source_subtoken_lengths)

        source_subtoken_indices = [[self.vocab.subtoken_vocab.lookup_index(x) for x in string]
                                   for string in source_subtoken_strings]
        source_subtoken_indices = torch.tensor(self._pad_sequence(source_subtoken_indices, pad_value=subtoken_pad_index,
                                               shape=(self.config.max_contexts, self.config.max_subtokens)))

        node_strings = [x[1].split('|') for x in split_contexts][:self.config.max_path_nodes]
        node_lengths = [len(string) for string in node_strings]
        node_lengths += [0 for _ in range(self.config.max_contexts - len(node_lengths))]
        node_lengths = torch.tensor(node_lengths)

        node_indices = [[self.vocab.node_vocab.lookup_index(x) for x in string] for string in node_strings]
        node_indices = torch.tensor(self._pad_sequence(node_indices, pad_value=node_pad_index,
                                    shape=(self.config.max_contexts, self.config.max_path_nodes)))

        target_subtoken_strings = [x[1].split('|') for x in split_contexts][:self.config.max_subtokens]
        target_subtoken_lengths = [len(string) for string in target_subtoken_strings]
        target_subtoken_lengths += [0 for _ in range(self.config.max_contexts - len(target_subtoken_strings))]
        target_subtoken_lengths = torch.tensor(target_subtoken_lengths)

        target_subtoken_indices = [[self.vocab.subtoken_vocab.lookup_index(x) for x in string]
                                   for string in target_subtoken_strings]
        target_subtoken_indices = torch.tensor(self._pad_sequence(target_subtoken_indices, pad_value=subtoken_pad_index,
                                               shape=(self.config.max_contexts, self.config.max_subtokens)))

        context_valid_mask = (torch.ne(torch.max(source_subtoken_indices, -1).values, subtoken_pad_index) |
                              torch.ne(torch.max(target_subtoken_indices, -1).values, subtoken_pad_index) |
                              torch.ne(torch.max(node_indices, -1).values, node_pad_index)).float()

        return SequentialPathContextInput(
            node_indices=node_indices,
            node_lengths=node_lengths,
            source_subtoken_indices=source_subtoken_indices,
            source_subtoken_lengths=source_subtoken_lengths,
            target_subtoken_indices=target_subtoken_indices,
            target_subtoken_lengths=target_subtoken_lengths,
            context_valid_mask=context_valid_mask,
            target_indices=target_indices,
            sample_index=torch.tensor(index)
        ), target_label

    def is_valid_input_row(self, input_tensor, split) -> bool:
        subtoken_pad_index = self.vocab.subtoken_vocab.word_to_index[self.vocab.subtoken_vocab.special_words.PAD]
        node_pad_index = self.vocab.node_vocab.word_to_index[self.vocab.node_vocab.special_words.PAD]

        any_context_is_valid = (torch.max(input_tensor.source_subtoken_indices).item() != subtoken_pad_index |
                                torch.max(input_tensor.target_subtoken_indices).item() != subtoken_pad_index |
                                torch.max(input_tensor.node_indices).item() != node_pad_index)

        if split.is_dev or split.is_test:
            return any_context_is_valid
        else:
            target_oov_index = self.vocab.target_vocab.word_to_index[self.vocab.target_vocab.special_words.OOV]
            word_is_valid = torch.max(input_tensor.target_indices) > target_oov_index
            return word_is_valid and any_context_is_valid

    @staticmethod
    def _pad_sequence(sequence, shape, pad_value=None):
        out = np.full(shape, pad_value)
        for i0 in range(min(shape[0], len(sequence))):
            for i1 in range(min(shape[1], len(sequence[i0]))):
                out[i0][i1] = sequence[i0][i1]
        return out
