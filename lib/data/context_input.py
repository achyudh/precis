from typing import NamedTuple

import torch


class PathContextInput(NamedTuple):
    path_indices: torch.Tensor
    source_token_indices: torch.Tensor
    target_token_indices: torch.Tensor
    context_valid_mask: torch.Tensor
    target_index: torch.Tensor
    sample_index: torch.Tensor


class SequentialPathContextInput(NamedTuple):
    node_indices: torch.Tensor
    node_lengths: torch.Tensor
    source_subtoken_indices: torch.Tensor
    source_subtoken_lengths: torch.Tensor
    target_subtoken_indices: torch.Tensor
    target_subtoken_lengths: torch.Tensor
    context_valid_mask: torch.Tensor
    target_indices: torch.Tensor
    sample_index: torch.Tensor