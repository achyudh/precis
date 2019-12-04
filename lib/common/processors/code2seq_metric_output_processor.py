import torch
import torch.nn.functional as F

from lib.common.processors.abstract_output_processor import AbstractOutputProcessor


class Code2SeqMetricOutputProcessor(AbstractOutputProcessor):
    def __init__(self, config, vocab, dataset):
        self.config = config
        self.vocab = vocab
        self.dataset = dataset
        self.target_sos_index = self.vocab.target_vocab.word_to_index[self.vocab.target_vocab.special_words.SOS]
        self.target_eos_index = self.vocab.target_vocab.word_to_index[self.vocab.target_vocab.special_words.PAD]

    def process(self, logits: torch.Tensor, sample_indices: torch.Tensor):
        target_strings = [self.dataset.target_labels[index.item()] for index in sample_indices]  # (batch,)

        logits = F.softmax(logits, dim=-1)  # (batch, max_target_len, target_vocab_size)
        top_k_indices = torch.topk(logits, self.config.top_k).indices  # (batch, max_target_len, k)
        top_k_indices = torch.transpose(top_k_indices, 1, 2)  # (batch, k, max_target_len)
        top_k_strings = [['|'.join(self.vocab.target_vocab.lookup_word(index.item()) for index in target_indices)
                         for target_indices in input_row] for input_row in top_k_indices]  # (batch, k)

        return list(zip(target_strings, top_k_strings))
