import random
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch

from lib.data import SemiSeqPathContextInput
from lib.data.readers.dataset_reader import DatasetReader
from lib.data.vocab import PathContextVocabContainer


class SeqPathContextReader(DatasetReader):
    def __init__(self, config: Namespace, vocab: PathContextVocabContainer):
        super().__init__(config, vocab)
        self.path_pad_index = vocab.path_vocab.word_to_index[vocab.path_vocab.special_words.PAD]
        self.target_eos_index = vocab.target_vocab.word_to_index[vocab.target_vocab.special_words.EOS]
        self.target_pad_index = vocab.target_vocab.word_to_index[vocab.target_vocab.special_words.PAD]
        self.token_pad_index = vocab.token_vocab.word_to_index[vocab.token_vocab.special_words.PAD]

    def is_valid_input_row(self, input_tensor, split) -> bool:
        any_context_is_valid = (torch.max(input_tensor.source_subtoken_indices).item() != self.token_pad_index |
                                torch.max(input_tensor.target_subtoken_indices).item() != self.token_pad_index |
                                torch.max(input_tensor.node_indices).item() != self.path_pad_index)

        if split.is_dev or split.is_test:
            return any_context_is_valid
        else:
            target_oov_index = self.vocab.target_vocab.word_to_index[self.vocab.target_vocab.special_words.OOV]
            word_is_valid = torch.max(input_tensor.label_indices) > target_oov_index
            return word_is_valid and any_context_is_valid

    def _get_input_tensors(self, index, *row_parts) -> Tuple:
        row_parts = list(row_parts)

        target_label = row_parts[0]
        label_strings = target_label.split('|')[:self.config.max_target_length - 1]
        label_strings = [self.vocab.target_vocab.lookup_index(x) for x in label_strings]

        label_strings.append(self.target_eos_index)
        label_strings.extend(self.target_pad_index for _ in range(self.config.max_target_length - len(label_strings)))
        label_indices = torch.tensor(label_strings)

        split_contexts = [x.split(',') for x in row_parts[1:]]

        if self.config.context_sampling:
            split_contexts = random.choices(split_contexts, k=self.config.max_contexts)
        else:
            split_contexts = split_contexts[:self.config.max_contexts]

        source_subtoken_strings = [x[0].split('|') for x in split_contexts][:self.config.max_subtokens]
        source_subtoken_indices, source_subtoken_lengths = self._process_subtoken_strings(source_subtoken_strings)

        node_strings = [x[1].split('|') for x in split_contexts][:self.config.max_path_nodes]
        node_lengths = [len(string) for string in node_strings]
        node_lengths += [1 for _ in range(self.config.max_contexts - len(node_lengths))]
        node_lengths = torch.tensor(node_lengths)

        node_indices = [[self.vocab.path_vocab.lookup_index(x) for x in string] for string in node_strings]
        node_indices = torch.tensor(self._pad_sequence(node_indices, pad_value=self.path_pad_index,
                                                       shape=(self.config.max_contexts, self.config.max_path_nodes)))

        target_subtoken_strings = [x[2].split('|') for x in split_contexts][:self.config.max_subtokens]
        target_subtoken_indices, target_subtoken_lengths = self._process_subtoken_strings(target_subtoken_strings)

        context_valid_mask = (torch.ne(torch.max(source_subtoken_indices, -1).values, self.token_pad_index) |
                              torch.ne(torch.max(target_subtoken_indices, -1).values, self.token_pad_index) |
                              torch.ne(torch.max(node_indices, -1).values, self.path_pad_index)).float()

        return SemiSeqPathContextInput(
            sample_index=torch.tensor(index),
            label_index=label_indices,
            node_indices=node_indices,
            node_lengths=node_lengths,
            source_subtoken_indices=source_subtoken_indices,
            source_subtoken_lengths=source_subtoken_lengths,
            target_subtoken_indices=target_subtoken_indices,
            target_subtoken_lengths=target_subtoken_lengths,
            context_valid_mask=context_valid_mask
        ), target_label

    def _process_subtoken_strings(self, subtoken_strings):
        subtoken_lengths = [len(string) for string in subtoken_strings]
        subtoken_lengths += [0 for _ in range(self.config.max_contexts - len(subtoken_strings))]
        subtoken_lengths = torch.tensor(subtoken_lengths)

        subtoken_indices = [[self.vocab.token_vocab.lookup_index(x) for x in string] for string in subtoken_strings]
        subtoken_indices = torch.tensor(self._pad_sequence(subtoken_indices, pad_value=self.token_pad_index,
                                                           shape=(self.config.max_contexts, self.config.max_subtokens)))

        return subtoken_indices, subtoken_lengths

    @staticmethod
    def _pad_sequence(sequence, shape, pad_value=None):
        out = np.full(shape, pad_value)
        for i0 in range(min(shape[0], len(sequence))):
            for i1 in range(min(shape[1], len(sequence[i0]))):
                out[i0][i1] = sequence[i0][i1]
        return out
