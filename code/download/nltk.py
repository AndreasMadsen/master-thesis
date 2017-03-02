
from typing import List

import nltk

from code.download.abstract.cache_dir import CacheDir


class NLTKEnv(CacheDir):
    _old_nltk_dir: List[str]

    def __init__(self) -> None:
        super().__init__(name='nltk')

    def __enter__(self):
        # set nltk path
        self._old_nltk_dir = nltk.data.path
        nltk.data.path = [self.dirpath]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore nltk path
        nltk.data.path = self._old_nltk_dir

    def download(self, name: str) -> None:
        if not self.exists(name):
            nltk.download(name, download_dir=self.nltk_dir)

    def exists(self, name: str) -> bool:
        try:
            getattr(nltk.corpus, name).ensure_loaded()
        except LookupError:
            return False
        return True

    def filepath(self, name: str) -> str:
        raise NotImplementedError('filepaths are not available for NLTK.')
