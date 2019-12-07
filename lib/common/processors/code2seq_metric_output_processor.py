import torch

from lib.common.processors.abstract_output_processor import AbstractOutputProcessor


class Code2SeqMetricOutputProcessor(AbstractOutputProcessor):
    def __init__(self, config, vocab, dataset):
        self.config = config
        self.vocab = vocab
        self.dataset = dataset
        self.target_sos_index = self.vocab.target_vocab.word_to_index[self.vocab.target_vocab.special_words.SOS]
        self.target_eos_index = self.vocab.target_vocab.word_to_index[self.vocab.target_vocab.special_words.PAD]

    def process(self, top_k_sequences, sample_indices: torch.Tensor):
        target_strings = [self.dataset.target_labels[index.item()] for index in sample_indices]  # (batch,)
        top_k_strings = [['|'.join(self.vocab.target_vocab.lookup_word(index.item()) for index in sequence.indices)
                          for sequence in input_row] for input_row in top_k_sequences]  # (batch, k)
        return list(zip(target_strings, top_k_strings))
