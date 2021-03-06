from argparse import Namespace
from typing import Tuple

import numpy as np
import torch

from lib.data import SemiSeqPathContextInput
from lib.data.readers.dataset_reader import DatasetReader
from lib.data.vocab import PathContextVocabContainer


class SemiSeqPathContextReader(DatasetReader):
    def __init__(self, config: Namespace, vocab: PathContextVocabContainer):
        super().__init__(config, vocab)
        self.node_pad_index = vocab.path_vocab.word_to_index[vocab.path_vocab.special_words.PAD]
        self.subtoken_pad_index = vocab.token_vocab.word_to_index[vocab.token_vocab.special_words.PAD]

    def is_valid_input_row(self, input_tensor, split) -> bool:
        any_context_is_valid = (torch.max(input_tensor.source_subtoken_indices).item() != self.subtoken_pad_index |
                                torch.max(input_tensor.target_subtoken_indices).item() != self.subtoken_pad_index |
                                torch.max(input_tensor.node_indices).item() != self.node_pad_index)

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

        split_contexts = [x.split(',') for x in row_parts[1: self.config.max_contexts + 1]]

        source_subtoken_strings = [x[0].split('|') for x in split_contexts][:self.config.max_subtokens]
        source_subtoken_indices, source_subtoken_lengths = self._process_subtoken_strings(source_subtoken_strings)

        node_strings = [x[1].split('|') for x in split_contexts][:self.config.max_path_nodes]
        node_lengths = [len(string) for string in node_strings]
        node_lengths += [1 for _ in range(self.config.max_contexts - len(node_lengths))]
        node_lengths = torch.tensor(node_lengths)

        node_indices = [[self.vocab.path_vocab.lookup_index(x) for x in string] for string in node_strings]
        node_indices = torch.tensor(self._pad_sequence(node_indices, pad_value=self.node_pad_index,
                                    shape=(self.config.max_contexts, self.config.max_path_nodes)))

        target_subtoken_strings = [x[2].split('|') for x in split_contexts][:self.config.max_subtokens]
        target_subtoken_indices, target_subtoken_lengths = self._process_subtoken_strings(target_subtoken_strings)

        context_valid_mask = (torch.ne(torch.max(source_subtoken_indices, -1).values, self.subtoken_pad_index) |
                              torch.ne(torch.max(target_subtoken_indices, -1).values, self.subtoken_pad_index) |
                              torch.ne(torch.max(node_indices, -1).values, self.node_pad_index)).float()

        return SemiSeqPathContextInput(
            node_indices=node_indices,
            node_lengths=node_lengths,
            source_subtoken_indices=source_subtoken_indices,
            source_subtoken_lengths=source_subtoken_lengths,
            target_subtoken_indices=target_subtoken_indices,
            target_subtoken_lengths=target_subtoken_lengths,
            context_valid_mask=context_valid_mask,
            label_index=label_index,
            sample_index=torch.tensor(index)
        ), label_string

    def _process_subtoken_strings(self, subtoken_strings):
        subtoken_lengths = [len(string) for string in subtoken_strings]
        subtoken_lengths += [0 for _ in range(self.config.max_contexts - len(subtoken_strings))]
        subtoken_lengths = torch.tensor(subtoken_lengths)

        subtoken_indices = [[self.vocab.token_vocab.lookup_index(x) for x in string] for string in subtoken_strings]
        subtoken_indices = torch.tensor(self._pad_sequence(subtoken_indices, pad_value=self.subtoken_pad_index,
                                        shape=(self.config.max_contexts, self.config.max_subtokens)))

        return subtoken_indices, subtoken_lengths

    @staticmethod
    def _pad_sequence(sequence, shape, pad_value=None):
        out = np.full(shape, pad_value)
        for i0 in range(min(shape[0], len(sequence))):
            for i1 in range(min(shape[1], len(sequence[i0]))):
                out[i0][i1] = sequence[i0][i1]
        return out
