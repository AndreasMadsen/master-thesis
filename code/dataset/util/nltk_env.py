
from typing import List
import os.path as path

import nltk

from code.dataset.util.download_dir import download_dir


class NLTKEnv:
    nltk_dir: str
    _old_dir: List[str]

    def __init__(self, nltk_dir: str=None) -> None:
        if nltk_dir is None:
            nltk_dir = path.join(download_dir(), 'nltk')

        self.nltk_dir = nltk_dir

    def __enter__(self):
        # set nltk path
        self._old_nltk = nltk.data.path
        nltk.data.path = [self.nltk_dir]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore nltk path
        nltk.data.path = self._old_nltk

    def download(self, name: str) -> None:
        # download `name` if it isn't already downloaded
        try:
            getattr(nltk.corpus, name).ensure_loaded()
        except LookupError:
            nltk.download(name, download_dir=self.nltk_dir)
