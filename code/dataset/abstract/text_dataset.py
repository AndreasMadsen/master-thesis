
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
    effective_max_length: int
    vocabulary_size: int
    vocabulary: FrozenSet[str]
    max_length: int
    decode: Mapping[int, str]
    encode: Mapping[str, int]
    encode_dtype: np.unsignedinteger

    def __init__(self,
                 vocabulary: FrozenSet[str]=None,
                 max_length: int=None,
                 **kwargs) -> None:

        # get corpus properties
        self.vocabulary = vocabulary
        if vocabulary is None:
            self.vocabulary = self._compute_vocabulary()

        self.max_length = max_length
        if max_length is None:
            self.max_length = self._compute_length()

        # validate properties
        if '#' in vocabulary:
            raise ValueError('a special char (#) was found in the vocabulary')
        if '_' in vocabulary:
            raise ValueError('a special char (_) aws found in the vocabulary')
        if max_length <= 0:
            raise ValueError('max_length must be positive')

        # create encoding schema
        self._setup_encoding()

        # extract sources and targets
        sources, targets = zip(*self)

        # encode source and targets
        sources = self.encode_as_batch(sources)
        targets = self.encode_as_batch(targets)

        # setup tensorflow pipeline
        super().__init__(sources, targets, **kwargs)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, str]]:
        pass

    @property
    def corpus_properties(self) -> CorpusProperties:
        return CorpusProperties(
            vocabulary=self.vocabulary,
            max_length=self.max_length
        )

    def _compute_vocabulary(self) -> FrozenSet[str]:
        unique_chars = set()
        for source, target in self:
            # add source and target to the char set
            unique_chars |= set(source)
            unique_chars |= set(target)

        return frozenset(unique_chars)

    def _compute_length(self) -> int:
        max_length = 0
        for source, target in self:
            # update max length
            max_length = max(max_length, len(source), len(target))

        return max_length

    def _setup_encoding(self) -> None:
        # set effective max length, (longest str + <EOS>)
        self.effective_max_length = self.max_length + 1

        # to ensure consistent encoding, sort the chars.
        # also add a null char for padding and and <EOS> char for EOS.
        self.decode = ['_', '#'] + sorted(self.vocabulary)

        # set vocabulary size
        self.vocabulary_size = len(self.decode)

        # reverse the decoder list to create an encoder map
        self.encode = {
            val: index for index, val in enumerate(self.decode)
        }

        # auto detect appropiate encoding type
        self.encode_dtype = size_to_signed_type(len(self.decode))

    def encode_as_batch(self, corpus: List[str]) -> np.ndarray:
        batch = np.empty(
            (len(corpus), self.effective_max_length),
            self.encode_dtype
        )

        for i, text in enumerate(corpus):
            batch[i] = self.encode_as_array(text)

        return batch

    def encode_as_iter(self, decoded: str) -> Iterator[int]:
        for char in decoded:
            yield self.encode[char]

        yield 1  # <EOS>

        for _ in range(0, self.effective_max_length - len(decoded) - 1):
            yield 0  # NULL

    def encode_as_array(self, decoded: str) -> np.ndarray:
        return np.fromiter(
            iter=self.encode_as_iter(decoded),
            dtype=self.encode_dtype,
            count=self.effective_max_length
        )

    def decode_as_str(self, encoded: np.ndarray, show_eos: bool=True) -> str:
        decoded = ''
        for code in encoded:
            if code != 1 or show_eos:
                decoded += self.decode[code]
            if code == 1:
                break

        return decoded

    def decode_as_batch(self, encoded: np.ndarray, **kwargs) -> List[str]:
        return [self.decode_as_str(row, **kwargs) for row in encoded]
