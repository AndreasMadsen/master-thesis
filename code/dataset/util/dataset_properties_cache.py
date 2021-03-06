
from typing import Any, Mapping, FrozenSet, NamedTuple

import os
import os.path as path
import json

from code.dataset.util.length_histogram import LengthHistogram

CorpusProperties = NamedTuple('CorpusProperties', [
    ('histogram', LengthHistogram),
    ('vocabulary', FrozenSet[str])
])

_this_dir = path.dirname(path.realpath(__file__))

_format_string = '''\
\t[
\t\t{key},
\t\t{{
\t\t\t"histogram": {histogram},
\t\t\t"vocabulary": {vocabulary}
\t\t}}
\t]{tail}'''


class DatasetCache:
    """Maneges the cache for all datasets"""
    _filepath: str

    _cache: Mapping[Any, CorpusProperties] = dict()

    def __init__(self, filepath: str):
        self._filepath = filepath

        if path.isfile(self._filepath):
            self._load_cache()

    def _load_cache(self):
        # load cache
        with open(self._filepath, 'r', encoding='utf-8') as fd:
            deserialized_cache = json.load(fd)
            self._cache = {
                tuple(key): CorpusProperties(
                    histogram=LengthHistogram(val['histogram']),
                    vocabulary=frozenset(val['vocabulary'])
                ) for key, val in deserialized_cache
            }

    def _save_cache(self):
        with open(self._filepath, 'w') as fd:
            print('[', file=fd)

            for i, (key, val) in enumerate(self._cache.items()):
                print(_format_string.format(
                    key=json.dumps(key),
                    histogram=json.dumps(val.histogram.encode()),
                    vocabulary=json.dumps(
                        list(val.vocabulary), ensure_ascii=False
                    ),
                    tail='' if i + 1 == len(self._cache) else ','
                ), file=fd)
            print(']', file=fd)

    def __getitem__(self, key: Any):
        return self._cache[key]

    def __setitem__(self, key: Any, val: CorpusProperties):
        self._cache[key] = val
        self._save_cache()

    def __contains__(self, key: Any):
        return key in self._cache


class PropertyCache:
    """Maneges the cache for all datasets"""
    _cachedir: str = path.realpath(path.join(_this_dir, '..', 'cache'))

    _cache: Mapping[str, DatasetCache] = dict()

    def __init__(self):
        # ensure `cachedir` exists
        try:
            os.mkdir(self._cachedir, mode=0o755)
        except FileExistsError:
            pass

    def __getitem__(self, key: str):
        if key not in self._cache:
            filepath = path.join(self._cachedir, f'{key}.json')
            self._cache[key] = DatasetCache(filepath)

        return self._cache[key]


property_cache = PropertyCache()
