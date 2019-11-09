from enum import Enum
from typing import NamedTuple, Optional

import torch

from lib.config import Config
from lib.data.vocab import Code2VecVocab


class EstimatorAction(Enum):
    Train = 'train'
    Evaluate = 'evaluate'
    Predict = 'predict'

    @property
    def is_train(self):
        return self is EstimatorAction.Train

    @property
    def is_evaluate(self):
        return self is EstimatorAction.Evaluate

    @property
    def is_predict(self):
        return self is EstimatorAction.Predict

    @property
    def is_evaluate_or_predict(self):
        return self.is_evaluate or self.is_predict


class ReaderInputTensors(NamedTuple):
    """
    Used mostly for convenient-and-clear access to input parts (by their names).
    """
    path_source_token_indices: torch.Tensor
    path_indices: torch.Tensor
    path_target_token_indices: torch.Tensor
    context_valid_mask: torch.Tensor
    target_index: Optional[torch.Tensor] = None
    target_string: Optional[torch.Tensor] = None
    path_strings: Optional[torch.Tensor] = None
    path_source_token_strings: Optional[torch.Tensor] = None
    path_target_token_strings: Optional[torch.Tensor] = None


class PathContextReader:
    def __init__(self,
                 vocabs: Code2VecVocab,
                 config: Config,
                 estimator_action: EstimatorAction):
        self.vocabs = vocabs
        self.config = config
        self.estimator_action = estimator_action

    def process_input_row(self, row_placeholder):
        parts = row_placeholder.split(' ')
        return self._map_raw_dataset_row_to_input_tensors(*parts)

    def is_valid_input_row(self, input_tensor) -> bool:
        token_pad_index = self.vocabs.token_vocab.word_to_index[self.vocabs.token_vocab.special_words.PAD]
        path_pad_index = self.vocabs.path_vocab.word_to_index[self.vocabs.path_vocab.special_words.PAD]

        any_contexts_is_valid = (torch.max(input_tensor.path_source_token_indices).item() != token_pad_index |
                                 torch.max(input_tensor.path_target_token_indices).item() != token_pad_index |
                                 torch.max(input_tensor.path_indices).item() != path_pad_index)

        if self.estimator_action.is_evaluate:
            return any_contexts_is_valid  # scalar
        else:
            target_oov_index = self.vocabs.target_vocab.word_to_index[self.vocabs.target_vocab.special_words.OOV]
            word_is_valid = input_tensor.target_index > target_oov_index
            return word_is_valid and any_contexts_is_valid

    def _map_raw_dataset_row_to_input_tensors(self, *row_parts) -> ReaderInputTensors:
        row_parts = list(row_parts)
        target_str = row_parts[0]
        target_index = torch.tensor(self.vocabs.target_vocab.lookup_index(target_str))

        split_contexts = [x.split(',') for x in row_parts[1: self.config.MAX_CONTEXTS + 1]]

        token_pad_string = self.vocabs.token_vocab.special_words.PAD
        path_pad_string = self.vocabs.path_vocab.special_words.PAD
        token_pad_index = self.vocabs.token_vocab.word_to_index[token_pad_string]
        path_pad_index = self.vocabs.path_vocab.word_to_index[path_pad_string]

        path_source_token_strings = torch.tensor([x[0] if len(x) > 0 else token_pad_string for x in split_contexts])
        path_strings = torch.tensor([x[1] if len(x) > 1 else path_pad_string for x in split_contexts])
        path_target_token_strings = torch.tensor([x[2] if len(x) > 2 else token_pad_string for x in split_contexts])

        path_source_token_indices = torch.tensor([self.vocabs.token_vocab.lookup_index(x[0]) for x in split_contexts])
        path_indices = torch.tensor([self.vocabs.token_vocab.lookup_index(x[1]) for x in split_contexts])
        path_target_token_indices = torch.tensor([self.vocabs.token_vocab.lookup_index(x[2]) for x in split_contexts])

        context_valid_mask = (torch.ne(path_source_token_indices, token_pad_index) |
                              torch.ne(path_target_token_indices, token_pad_index) |
                              torch.ne(path_indices, path_pad_index)).float()

        return ReaderInputTensors(
            path_source_token_indices=path_source_token_indices,
            path_indices=path_indices,
            path_target_token_indices=path_target_token_indices,
            context_valid_mask=context_valid_mask,
            target_index=target_index,
            target_string=target_str,
            path_source_token_strings=path_source_token_strings,
            path_strings=path_strings,
            path_target_token_strings=path_target_token_strings
        )