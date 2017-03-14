
from typing import Any, List, Tuple, Mapping, Iterator, FrozenSet
import abc

import numpy as np
from tqdm import tqdm as tqdm_bar

from code.dataset.abstract.dataset import Dataset
from code.dataset.util.size_to_type import size_to_signed_type
from code.dataset.util.length_histogram import LengthHistogram
from code.dataset.util.dataset_properties_cache import \
    CorpusProperties, property_cache


class TextDataset(Dataset):
    properties: CorpusProperties
    _decode: Mapping[int, str]
    _encode: Mapping[str, int]
    _encode_dtype: np.unsignedinteger
    _show_tqdm: bool = False
    source_lang: str
    target_lang: str

    def __init__(self,
                 source_lang: str, target_lang: str,
                 key: Any=None,
                 vocabulary: FrozenSet[str]=None,
                 observations: int=None,
                 validate: bool=False,
                 name: str='unamed',
                 tqdm: bool=True,
                 **kwargs) -> None:
        # save basic properties
        self._show_tqdm = tqdm

        # compute properties
        computed_properties = self._fetch_corpus_properties(
            name, key, observations=observations
        )
        # use computed vocabulary if not provided
        if vocabulary is None:
            vocabulary = computed_properties.vocabulary

        # make properties public
        self.properties = CorpusProperties(
            vocabulary=vocabulary,
            histogram=computed_properties.histogram
        )

        # validate properties
        # http://unicode-search.net/unicode-namesearch.pl
        if '^' in vocabulary:
            raise ValueError('a eos char (^) was found in the vocabulary')
        if '⨯' in vocabulary:
            raise ValueError('a null char (⨯) was found in the vocabulary')
        if '�' in vocabulary:
            raise ValueError('an invalid char (�) was found in the vocabulary')

        # create encoding schema
        self._setup_encoding()

        # set language properties
        self.source_lang = source_lang
        self.target_lang = target_lang

        # validate corpus properties
        if validate:
            self._validate_corpus_properties(name)

        # setup tensorflow pipeline
        super().__init__(histogram=self.properties.histogram,
                         dtype=self._encode_dtype,
                         name=name,
                         tqdm=tqdm,
                         **kwargs)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, str]]:
        pass

    @property
    def vocabulary_size(self) -> int:
        # add <eos> <null> symbols
        return len(self.properties.vocabulary) + 2

    @property
    def vocabulary(self) -> FrozenSet[str]:
        return self.properties.vocabulary

    @property
    def labels(self) -> Mapping[int, str]:
        return self._decode

    def _fetch_corpus_properties(self,
                                 name: str, key: Any,
                                 observations: int=None) -> CorpusProperties:
        # don't involve the cache if key is None
        if key is None:
            return self._compute_corpus_properties(expected_obs=observations)

        # build the cache if not already build
        if key not in property_cache[name]:
            property_cache[name][key] = self._compute_corpus_properties(
                expected_obs=observations
            )

        return property_cache[name][key]

    def _compute_corpus_properties(self,
                                   tqdm_message: str='corpus properties',
                                   expected_obs: int=None) -> CorpusProperties:
        unique_chars = set()
        histogram = LengthHistogram()

        for source, target in tqdm_bar(self,
                                       total=expected_obs,
                                       unit='obs', desc=tqdm_message,
                                       disable=not self._show_tqdm):
            # add source and target to the char set
            unique_chars |= set(source)
            unique_chars |= set(target)

            # source and target will be padded to equal length
            length = max(len(source), len(target))
            histogram.add(length)

        return CorpusProperties(
            vocabulary=unique_chars,
            histogram=histogram
        )

    def _validate_corpus_properties(self, name: str) -> None:
        truth = self._compute_corpus_properties(
            tqdm_message='validate properties',
            expected_obs=self.properties.histogram.observations
        )
        properties_valid = True

        print(f'Dataset validation ({name}):')
        cache_hist = self.properties.histogram
        truth_hist = truth.histogram

        if cache_hist.observations != truth_hist.observations:
            properties_valid = False

            print(f'  The observations count is wrong:')
            print(f'  {cache_hist.observations} != {truth_hist.observations}')
            print(f'  The behaviour is undefined.')

        if cache_hist.max_length != truth_hist.max_length:
            properties_valid = False

            print(f'  The max-length value is wrong:')
            print(f'  {cache_hist.max_length} != {truth_hist.max_length}')
            print(f'  This may not be an issue.')

        if cache_hist.min_length != truth_hist.min_length:
            properties_valid = False

            print(f'  The min-length value is wrong:')
            print(f'  {cache_hist.min_length} != {truth_hist.min_length}')
            print(f'  This may not be an issue.')

        if cache_hist.encode() != truth_hist.encode():
            properties_valid = False

            print(f'  The histogram does not match:')
            print(f'  {cache_hist.encode()} != {truth_hist.encode()}')
            print(f'  This may not be an issue.')

        if len(truth.vocabulary - self.properties.vocabulary) > 0:
            properties_valid = False

            missing_chars = truth.vocabulary - self.properties.vocabulary
            print(f'  The following chars was not found in the vocabulary:')
            print(f'  {{{", ".join(sorted(missing_chars))}}}')
            print(f'  Missing characters will be ignored.')

        if properties_valid:
            print('  The corpus properties are valid.')

    def _setup_encoding(self) -> None:
        # to ensure consistent encoding, sort the chars.
        # also add a null char for padding and and <EOS> char for EOS.
        self._decode = ['⨯', '^'] + sorted(self.vocabulary)

        # reverse the decoder list to create an encoder map
        self._encode = {
            val: index for index, val in enumerate(self._decode)
        }

        # auto detect appropiate encoding type
        self._encode_dtype = size_to_signed_type(self.vocabulary_size)

    def encode_as_batch(self, corpus: Iterator[str]) -> np.ndarray:
        max_length = 0
        observations = 0

        for i, text in enumerate(corpus):
            # include <EOS> in the length
            max_length = max(max_length, len(text) + 1)
            observations += 1

        batch = np.empty(
            (observations, max_length),
            self._encode_dtype
        )

        for i, text in enumerate(corpus):
            datum = self.encode_as_array(text)
            batch[i] = np.pad(datum, (0, max_length - len(datum)), 'constant')

        return batch

    def encode_as_iter(self, decoded: str) -> Iterator[int]:
        length = 0
        for char in decoded:
            if char in self._encode:
                yield self._encode[char]
                length += 1

        yield 1  # <EOS>

    def encode_as_array(self, decoded: str) -> np.ndarray:
        return np.fromiter(
            iter=self.encode_as_iter(decoded),
            dtype=self._encode_dtype
        )

    def decode_as_str(self, encoded: np.ndarray, show_eos: bool=True) -> str:
        decoded = ''
        for code in encoded:
            if code >= self.vocabulary_size:
                decoded += '�'
            elif code != 1 or show_eos:
                decoded += self._decode[code]

            if code == 1:
                break

        return decoded

    def decode_as_batch(self, encoded: np.ndarray, **kwargs) -> List[str]:
        return [self.decode_as_str(row, **kwargs) for row in encoded]
