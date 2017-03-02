
from code.download.abstract.cache_dir import CacheDir


class WMTCache(CacheDir):
    def __init__(self) -> None:
        super().__init__(name='wmt')


class EuroparlCache(CacheDir):
    def __init__(self) -> None:
        super().__init__(name='europarl')
