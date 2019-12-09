from typing import NamedTuple

import torch


class PathContextInput(NamedTuple):
    sample_index: torch.Tensor
    label_index: torch.Tensor
    path_indices: torch.Tensor
    source_token_indices: torch.Tensor
    target_token_indices: torch.Tensor
    context_valid_mask: torch.Tensor


class SemiSeqPathContextInput(NamedTuple):
    sample_index: torch.Tensor
    label_index: torch.Tensor
    node_indices: torch.Tensor
    node_lengths: torch.Tensor
    source_subtoken_indices: torch.Tensor
    source_subtoken_lengths: torch.Tensor
    target_subtoken_indices: torch.Tensor
    target_subtoken_lengths: torch.Tensor
    context_valid_mask: torch.Tensor


class SeqPathContextInput(NamedTuple):
    sample_index: torch.Tensor
    label_indices: torch.Tensor
    node_indices: torch.Tensor
    node_lengths: torch.Tensor
    source_subtoken_indices: torch.Tensor
    source_subtoken_lengths: torch.Tensor
    target_subtoken_indices: torch.Tensor
    target_subtoken_lengths: torch.Tensor
    context_valid_mask: torch.Tensor
