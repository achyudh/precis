from collections import Counter

import numpy as np

from lib.util.preprocessing import filter_impossible_names, get_subtokens


class SubtokenCompositionMetric:
    def __init__(self, special_words):
        self.num_true_positives: int = 0
        self.num_false_positives: int = 0
        self.num_false_negatives: int = 0
        self.num_predictions: int = 0
        self.special_words = special_words

    def update_batch(self, results):
        for original_name, top_words in results:
            prediction = filter_impossible_names(self.special_words, top_words)
            if prediction:
                original_subtokens = Counter(get_subtokens(original_name))
                predicted_subtokens = Counter(get_subtokens(prediction[0]))
                self.num_true_positives += sum(count for element, count in predicted_subtokens.items()
                                               if element in original_subtokens)
                self.num_false_positives += sum(count for element, count in predicted_subtokens.items()
                                                if element not in original_subtokens)
                self.num_false_negatives += sum(count for element, count in original_subtokens.items()
                                                if element not in predicted_subtokens)
            self.num_predictions += 1

    @property
    def true_positives(self):
        return np.float64(self.num_true_positives) / self.num_predictions

    @property
    def false_positives(self):
        return np.float64(self.num_false_positives) / self.num_predictions

    @property
    def false_negatives(self):
        return np.float64(self.num_false_negatives) / self.num_predictions

    @property
    def precision(self):
        return np.float64(self.num_true_positives) / (self.num_true_positives + self.num_false_positives)

    @property
    def recall(self):
        return np.float64(self.num_true_positives) / (self.num_true_positives + self.num_false_negatives)

    @property
    def f1(self):
        return np.float64(2 * self.precision * self.recall) / (self.precision + self.recall)
