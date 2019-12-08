import torch

from lib.common.processors.output_processor import OutputProcessor
from lib.data.vocab.path_context_vocab_container import PathContextVocabContainer


class Code2VecMetricOutputProcessor(OutputProcessor):
    def __init__(self, k: int, vocab: PathContextVocabContainer, dataset):
        self.k = k
        self.vocab = vocab
        self.dataset = dataset

    def process(self, logits: torch.Tensor, sample_indices: torch.Tensor):
        top_k_indices = torch.topk(logits, self.k).indices
        top_k_strings = [[self.vocab.target_vocab.lookup_word(index.item()) for index in input_row]
                         for input_row in top_k_indices]
        target_strings = [self.dataset.target_labels[index.item()] for index in sample_indices]
        return list(zip(target_strings, top_k_strings))
