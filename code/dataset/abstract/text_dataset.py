
from typing import List, Tuple, Mapping, Iterator, FrozenSet, NamedTuple
import abc

import numpy as np

from code.dataset.abstract.dataset import Dataset
from code.dataset.util.size_to_type import size_to_signed_type

CorpusProperties = NamedTuple('CorpusProperties', [
    ('vocabulary', FrozenSet[str]),
    ('max_length', int)
])


class TextDataset(Dataset):
    vocabulary_size: int
    vocabulary: FrozenSet[str]
    max_length: int
    _decode: Mapping[int, str]
    _encode: Mapping[str, int]
    _encode_dtype: np.unsignedinteger
    source_lang: str
    target_lang: str

    def __init__(self,
                 source_lang, target_lang,
                 vocabulary: FrozenSet[str]=None,
                 max_length: int=None,
                 validate: bool=False,
                 name: str='unamed',
                 **kwargs) -> None:

        # compute properties if necessary
        if vocabulary is None or max_length is None:
            computed_vocabulary, computed_max_length = \
                self._compute_corpus_properties()

            if vocabulary is None:
                vocabulary = computed_vocabulary
            if max_length is None:
                max_length = computed_max_length

        # increment max_length such it includes eos
        self.vocabulary = vocabulary
        self.max_length = max_length + 1

        # validate properties
        if '^' in vocabulary:
            raise ValueError('a special char (^) was found in the vocabulary')
        if '_' in vocabulary:
            raise ValueError('a special char (_) was found in the vocabulary')
        if '~' in vocabulary:
            raise ValueError('a special char (~) was found in the vocabulary')
        if max_length <= 0:
            raise ValueError('max_length must be positive')

        # create encoding schema
        self._setup_encoding()

        # set language properties
        self.source_lang = source_lang
        self.target_lang = target_lang

        # extract sources and targets
        sources, targets = zip(*self)

        # validate vocabulary
        if validate:
            missing_chars = self._detect_missing_chars(sources)
            missing_chars |= self._detect_missing_chars(targets)
            print(f'Dataset validation ({name}):')
            if len(missing_chars):
                print('  The following chars was not found in the vocabulary:')
                print('  {' + ', '.join(sorted(missing_chars)) + '}')
                print('  Missing characters will be ignored.')
            else:
                print('  The vocabulary was complete.')

        # encode source and targets
        sources = self.encode_as_batch(sources)
        targets = self.encode_as_batch(targets)

        # setup tensorflow pipeline
        super().__init__(sources, targets, name=name, **kwargs)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, str]]:
        pass

    @property
    def corpus_properties(self) -> CorpusProperties:
        return CorpusProperties(
            vocabulary=self.vocabulary,
            max_length=self.max_length
        )

    @property
    def labels(self) -> Mapping[int, str]:
        return self._decode

    def _compute_corpus_properties(self) -> Tuple[FrozenSet[str], int]:
        max_length = 0
        unique_chars = set()

        for source, target in self:
            # add source and target to the char set
            unique_chars |= set(source)
            unique_chars |= set(target)

            # update max length
            max_length = max(max_length, len(source), len(target))

        return (unique_chars, max_length)

    def _setup_encoding(self) -> None:
        # to ensure consistent encoding, sort the chars.
        # also add a null char for padding and and <EOS> char for EOS.
        self._decode = ['_', '^'] + sorted(self.vocabulary)

        # set vocabulary size
        self.vocabulary_size = len(self._decode)

        # reverse the decoder list to create an encoder map
        self._encode = {
            val: index for index, val in enumerate(self._decode)
        }

        # auto detect appropiate encoding type
        self._encode_dtype = size_to_signed_type(self.vocabulary_size)

    def _detect_missing_chars(self, corpus: List[str]) -> FrozenSet[str]:
        missing_chars = set()

        for text in corpus:
            for char in text:
                if char not in self._encode:
                    missing_chars.add(char)

        return frozenset(missing_chars)

    def encode_as_batch(self, corpus: List[str]) -> np.ndarray:
        batch = np.empty(
            (len(corpus), self.max_length),
            self._encode_dtype
        )

        for i, text in enumerate(corpus):
            batch[i] = self.encode_as_array(text)

        return batch

    def encode_as_iter(self, decoded: str) -> Iterator[int]:
        length = 0
        for char in decoded:
            if char in self._encode:
                yield self._encode[char]
                length += 1

        yield 1  # <EOS>

        for _ in range(0, self.max_length - length - 1):
            yield 0  # NULL

    def encode_as_array(self, decoded: str) -> np.ndarray:
        return np.fromiter(
            iter=self.encode_as_iter(decoded),
            dtype=self._encode_dtype,
            count=self.max_length
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
