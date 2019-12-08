from argparse import Namespace
from enum import Enum


class VocabType(Enum):
    Token = 1
    Target = 2
    Path = 3
    Sequence = 4


_SpecialVocabWords_OnlyOov = Namespace(OOV='<OOV>')
_SpecialVocabWords_OovAndPad = Namespace(PAD='<PAD>', OOV='<OOV>')
_SpecialVocabWords_OovPadSeq = Namespace(PAD='<PAD>', OOV='<OOV>', SOS='<SOS>', EOS='<EOS>')


def get_special_words(vocab_type: VocabType) -> Namespace:
    if vocab_type == VocabType.Target:
        return _SpecialVocabWords_OnlyOov
    elif vocab_type == VocabType.Sequence:
        return _SpecialVocabWords_OovPadSeq
    else:
        return _SpecialVocabWords_OovAndPad