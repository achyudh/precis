import numpy as np

from lib.util.preprocessing import get_first_match_word_from_top_k


class TopKAccuracyMetric:
    def __init__(self, k: int, special_words):
        self.k = k
        self.special_words = special_words
        self.nr_predictions: int = 0
        self.nr_correct_predictions = np.zeros(self.k)

    def update_batch(self, results):
        for original_name, top_predicted_words in results:
            self.nr_predictions += 1
            found_match = get_first_match_word_from_top_k(self.special_words, original_name, top_predicted_words)
            if found_match is not None:
                suggestion_idx, _ = found_match
                self.nr_correct_predictions[suggestion_idx:self.k] += 1

    @property
    def top_k_accuracy(self):
        return self.nr_correct_predictions / self.nr_predictions
