
from typing import Iterator, List, Tuple

from code.download import NLTKEnv
from code.dataset.abstract.text_dataset import TextDataset


_bilingual_noswap = {
    ('en', 'fr'),
    ('de', 'fr'),
    ('de', 'en')
}

_bilingual_swap = {
    (target, source) for source, target in _bilingual_noswap
}


class NLTKComtrans(TextDataset):
    _source_lang: str
    _target_lang: str
    _min_length: int
    _max_length: int

    def __init__(self,
                 source_lang: str='fr',
                 target_lang: str='en',
                 min_length: int=50, max_length: int=150,
                 **kwargs) -> None:

        self._source_lang = source_lang
        self._target_lang = target_lang
        self._min_length = min_length
        self._max_length = max_length

        super().__init__(source_lang, target_lang,
                         name='nltk-comtrans',
                         **kwargs)

    def _comtrans_string(self) -> str:
        source_lang = self._source_lang
        target_lang = self._target_lang

        if (source_lang, target_lang) in _bilingual_swap:
            source_lang, target_lang = (target_lang, source_lang)

        return f'alignment-{source_lang}-{target_lang}.txt'

    def _comtrans_maybe_swap(
            self,
            als: Iterator[Tuple[List[str], List[str]]]
    ) -> Iterator[Tuple[List[str], List[str]]]:
        should_swap = (self._source_lang, self._target_lang) in _bilingual_swap

        for al in als:
            yield (al.mots, al.words) if should_swap else (al.words, al.mots)

    def __iter__(self) -> Iterator[Tuple[str, str]]:

        with NLTKEnv() as nltk_env:
            nltk_env.download('perluniprops')
            nltk_env.download('comtrans')

            from nltk.corpus import comtrans
            from nltk.tokenize.moses import MosesDetokenizer

            als = comtrans.aligned_sents(self._comtrans_string())

            source_detokenizer = MosesDetokenizer(lang=self._source_lang)
            target_detokenizer = MosesDetokenizer(lang=self._target_lang)

        for source, target in self._comtrans_maybe_swap(als):
            source = source_detokenizer.detokenize(source, return_str=True)
            target = target_detokenizer.detokenize(target, return_str=True)

            if self._min_length <= len(source) < self._max_length and \
               self._min_length <= len(target) < self._max_length:
                yield (source, target)
