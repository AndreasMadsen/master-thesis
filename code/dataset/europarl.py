
from typing import Optional, Iterator, Tuple
from contextlib import contextmanager
import tarfile

from code.download import EuroparlCache
from code.dataset.abstract.text_dataset import TextDataset
from code.dataset.util.length_checker import LengthChecker

_v7_url = 'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz'

_bilingual_noswap = {
    ('cs', 'en'): 646605,
    ('de', 'en'): 1920209,
    ('es', 'en'): 1965734,
    ('fr', 'en'): 2007723
}

_bilingual_swap = {
    (target, source): observations
    for (source, target), observations in _bilingual_noswap
}


@contextmanager
def tar_extract_file(tar_filepath, content_filepath):
    # The | sign means the tar file is opened in streaming mode,
    # this means that we need to iterate over all files and call
    # extractfile on the iterated file handle.
    # This trick is esential for performance and gives at least a 100x
    # performance boost, when reading two files simultaneously.
    with tarfile.open(tar_filepath, 'r|gz', encoding='utf-8') as target_tar:
        for t in target_tar:
            if t.name == content_filepath:
                yield (
                    line.decode('utf-8') for line in target_tar.extractfile(t)
                )
                break


class Europarl(TextDataset):
    _length_checker: LengthChecker
    _source_lang: str
    _target_lang: str
    _all_observations: bool = False

    def __init__(self,
                 source_lang: str='fr',
                 target_lang: str='en',
                 min_length: Optional[int]=50, max_length: Optional[int]=150,
                 **kwargs) -> None:
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._length_checker = LengthChecker(min_length, max_length)

        if min_length is None and max_length is None:
            self._all_observations = True

        super().__init__(
            source_lang, target_lang,
            observations=self._observation(),
            key=(source_lang, target_lang, min_length, max_length),
            name='europarl',
            **kwargs
        )

    def _observation(self) -> Optional[int]:
        if not self._all_observations:
            return None

        if (self._source_lang, self._target_lang) in _bilingual_noswap:
            return _bilingual_noswap[self._source_lang, self._target_lang]
        else:
            return _bilingual_swap[self._source_lang, self._target_lang]

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
            source_filepath, target_filepath = self._files()
            with tar_extract_file(filepath, source_filepath) as source_file, \
                    tar_extract_file(filepath, target_filepath) as target_file:

                for source, target in zip(source_file, target_file):
                    if self._length_checker(source, target):
                        yield (source, target)
