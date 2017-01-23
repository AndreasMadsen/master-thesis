
from typing import List, Tuple, Mapping, Iterator, FrozenSet
import abc

import numpy as np

from dataset.abstract.dataset import Dataset
from dataset.util.size_to_type import size_to_signed_type


class TextDataset(Dataset):
    effective_max_length: int
    vocabulary_size: int
    decode: Mapping[int, str]
    encode: Mapping[str, int]
    encode_dtype: np.unsignedinteger

    def __init__(self, vocabulary: FrozenSet[str]=None, **kwargs) -> None:
        # create encoding schema
        self._setup_encoding(vocabulary=vocabulary)

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

    def _setup_encoding(self, vocabulary: FrozenSet[str]=None) -> None:

        # find all unique chars and effective max length
        max_length = 0
        unique_chars = set() if (vocabulary is None) else vocabulary
        for source, target in self:
            if vocabulary is None:
                # add source and target to the char set
                unique_chars |= set(source)
                unique_chars |= set(target)

            # update max length
            max_length = max(max_length, len(source), len(target))

        # set effective max length, (longest str + <EOS>)
        self.effective_max_length = max_length + 1

        # to ensure consistent encoding, sort the chars.
        # also add a null char for padding and and <EOS> char for EOS.
        self.decode = ['_', '#'] + sorted(unique_chars)

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
