import torch

from lib.common.processors.abstract_output_processor import AbstractOutputProcessor
from lib.data.vocab import Code2VecVocabContainer


class Code2VecMetricOutputProcessor(AbstractOutputProcessor):
    def __init__(self, k: int, vocab: Code2VecVocabContainer, dataset):
        self.k = k
        self.vocab = vocab
        self.dataset = dataset

    def process(self, logits: torch.Tensor, sample_indices: torch.Tensor):
        top_k_indices = torch.topk(logits, self.k).indices
        top_k_strings = [[self.vocab.target_vocab.lookup_word(index.item()) for index in input_row]
                         for input_row in top_k_indices]
        target_strings = [self.dataset.target_labels[index.item()] for index in sample_indices]
        return list(zip(target_strings, top_k_strings))
