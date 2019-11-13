import re
from typing import Optional, Tuple


def get_subtokens(str):
    return str.split('|')


def normalize_word(word):
    stripped = re.sub(r'[^a-zA-Z]', '', word)
    if len(stripped) == 0:
        return word.lower()
    else:
        return stripped.lower()


def legal_method_names_checker(special_words, name):
    return name != special_words.OOV and re.match(r'^[a-zA-Z|]+$', name)


def filter_impossible_names(special_words, top_words):
    result = list(filter(lambda word: legal_method_names_checker(special_words, word), top_words))
    return result


def get_first_match_word_from_top_k(special_words, original_name, top_predicted_words) -> Optional[Tuple[int, str]]:
    normalized_original_name = normalize_word(original_name)
    for suggestion_idx, predicted_word in enumerate(filter_impossible_names(special_words, top_predicted_words)):
        normalized_possible_suggestion = normalize_word(predicted_word)
        if normalized_original_name == normalized_possible_suggestion:
            return suggestion_idx, predicted_word
    return None
