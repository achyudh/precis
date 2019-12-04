from typing import Optional, Tuple


def get_subtokens(str):
    return str.split('|')


def filter_impossible_names(special_words, top_words):
    result = list(filter(lambda word: word not in special_words, top_words))
    return result


def get_first_match_word_from_top_k(special_words, original_name, top_predicted_words) -> Optional[Tuple[int, str]]:
    normalized_original_name = original_name.lower()
    for suggestion_idx, predicted_word in enumerate(filter_impossible_names(special_words, top_predicted_words)):
        normalized_possible_suggestion = predicted_word.lower()
        if normalized_original_name == normalized_possible_suggestion:
            return suggestion_idx, predicted_word
    return None
