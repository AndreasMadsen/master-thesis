
from typing import Any, List, Tuple, Mapping, Iterator, FrozenSet
import abc

import numpy as np
from tqdm import tqdm

from code.dataset.abstract.dataset import Dataset
from code.dataset.util.size_to_type import size_to_signed_type
from code.dataset.util.dataset_properties_cache import \
    CorpusProperties, property_cache


class TextDataset(Dataset):
    properties: CorpusProperties
    _decode: Mapping[int, str]
    _encode: Mapping[str, int]
    _encode_dtype: np.unsignedinteger
    source_lang: str
    target_lang: str

    def __init__(self,
                 source_lang: str, target_lang: str,
                 key: Any=None,
                 vocabulary: FrozenSet[str]=None,
                 observations: int=None,
                 validate: bool=False,
                 name: str='unamed',
                 **kwargs) -> None:

        # compute properties if necessary
        if vocabulary is None or observations is None:
            computed_properties = self._fetch_corpus_properties(
                name, key, observations=observations
            )

            if vocabulary is None:
                vocabulary = computed_properties.vocabulary
            if observations is None:
                observations = computed_properties.observations

        # increment max_length such it includes eos
        self.properties = CorpusProperties(
            vocabulary=vocabulary,
            observations=observations
        )

        # validate properties
        if '^' in vocabulary:
            raise ValueError('a special char (^) was found in the vocabulary')
        if '_' in vocabulary:
            raise ValueError('a special char (_) was found in the vocabulary')
        if '~' in vocabulary:
            raise ValueError('a special char (~) was found in the vocabulary')

        # create encoding schema
        self._setup_encoding()

        # set language properties
        self.source_lang = source_lang
        self.target_lang = target_lang

        # validate corpus properties
        if validate:
            self._validate_corpus_properties()

        # setup tensorflow pipeline
        super().__init__(observations=observations,
                         dtype=self._encode_dtype,
                         name=name,
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
                                   expected_obs: int=None) -> CorpusProperties:
        unique_chars = set()
        observations = 0

        for source, target in tqdm(self,
                                   total=expected_obs,
                                   unit='obs', desc='corpus properties'):
            # add source and target to the char set
            unique_chars |= set(source)
            unique_chars |= set(target)

            # increment observations
            observations += 1

        return CorpusProperties(
            vocabulary=unique_chars,
            observations=observations
        )

    def _validate_corpus_properties(self) -> None:
        truth = self._compute_corpus_properties(
            expected_obs=self.properties.observations
        )
        properties_valid = True

        print(f'Dataset validation ({name}):')

        if len(truth.vocabulary - self.properties.vocabulary) > 0:
            properties_valid = False

            missing_chars = truth.vocabulary - self.properties.vocabulary
            print(f'  The following chars was not found in the vocabulary:')
            print(f'  {{{", ".join(sorted(missing_chars))}}}')
            print(f'  Missing characters will be ignored.')

        if self.properties.observations != truth.observations:
            properties_valid = False

            print(f'  The observations count is wrong:')
            print(f'  {self.properties.observations} != {truth.observations}')
            print(f'  The behaviour is undefined.')

        if properties_valid:
            print('  The corpus properties are valid.')

    def _setup_encoding(self) -> None:
        # to ensure consistent encoding, sort the chars.
        # also add a null char for padding and and <EOS> char for EOS.
        self._decode = ['_', '^'] + sorted(self.vocabulary)

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
                decoded += '~'
            elif code != 1 or show_eos:
                decoded += self._decode[code]

            if code == 1:
                break

        return decoded

    def decode_as_batch(self, encoded: np.ndarray, **kwargs) -> List[str]:
        return [self.decode_as_str(row, **kwargs) for row in encoded]
