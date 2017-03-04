
from typing import Iterator, Tuple
import tarfile

from code.download import EuroparlCache
from code.dataset.abstract.text_dataset import TextDataset

_v7_url = 'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz'

_bilingual_noswap = {
    ('cs', 'en'),
    ('de', 'en'),
    ('es', 'en'),
    ('fr', 'en')
}

_bilingual_swap = {
    (target, source) for source, target in _bilingual_noswap
}


class Europarl(TextDataset):
    _min_length: int
    _max_length: int
    _source_lang: str
    _target_lang: str

    def __init__(self,
                 source_lang: str='fr',
                 target_lang: str='en',
                 min_length: int=50, max_length: int=150,
                 **kwargs) -> None:
        self._min_length = min_length
        self._max_length = max_length
        self._source_lang = source_lang
        self._target_lang = target_lang

        super().__init__(
            source_lang, target_lang,
            key=(source_lang, target_lang, min_length, max_length),
            name='europarl',
            **kwargs
        )

    def _files(self) -> str:
        if (self._source_lang, self._target_lang) in _bilingual_swap:
            prefix = f'europarl-v7.{self._target_lang}-{self._source_lang}'
        else:
            prefix = f'europarl-v7.{self._source_lang}-{self._target_lang}'

        source_file = f'training/{prefix}.{self._source_lang}'
        target_file = f'training/{prefix}.{self._target_lang}'

        return (source_file, target_file)

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        with EuroparlCache() as europarl_cache:
            europarl_cache.download(name='europarl-v7.tgz', url=_v7_url)

            # peak in tarball
            filepath = europarl_cache.filepath('europarl-v7.tgz')
            with tarfile.open(filepath, 'r:gz', encoding='utf-8') as tar:
                # extract text files
                source_filepath, target_filepath = self._files()
                source_file = tar.extractfile(source_filepath)
                target_file = tar.extractfile(target_filepath)

                for source, target in zip(source_file, target_file):
                    # check string size
                    if self._min_length <= len(source) < self._max_length and \
                       self._min_length <= len(target) < self._max_length:
                        yield (source, target)
