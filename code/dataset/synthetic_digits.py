
from typing import Iterator, Tuple, FrozenSet

import numpy as np

from code.dataset.abstract.text_dataset import TextDataset
from code.dataset.util.size_to_type import size_to_unsigned_type

text_map = np.asarray([
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine'
])


class SyntheticDigits(TextDataset):
    _examples: int
    _digits: int
    _min_length: int
    _max_length: int
    _random: np.random.RandomState
    _dataset: Tuple[Tuple[str, str]]

    def __init__(self, examples: int=100,
                 digits: int=10,
                 min_length: int=2, max_length: int=3,
                 seed: int=None,
                 **kwargs) -> None:

        self._examples = examples
        self._digits = digits
        self._min_length = min_length
        self._max_length = max_length
        self._random = np.random.RandomState(seed)

        self._dataset = tuple(self._build_dataset())

        super().__init__(
            vocabulary=self._build_vocabulary(),
            seed=seed, **kwargs
        )

    def _build_vocabulary(self) -> FrozenSet[str]:
        source = ''.join(text_map)
        source_special = ' '
        target = ''.join(np.arange(0, self._digits).astype(np.str))
        return frozenset(source + source_special + target)

    def _build_dataset(self) -> Iterator[Tuple[str, str]]:
        length_type = size_to_unsigned_type(self._max_length)

        for _ in range(self._examples):
            length = self._random.randint(
                self._min_length, self._max_length + 1,
                dtype=length_type
            )

            target = self._random.randint(
                0, self._digits,
                size=length, dtype=np.int8
            )
            source = text_map[target]

            target_str = ''.join(target.astype(np.str))
            source_str = ' '.join(source)

            yield (source_str, target_str)

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return iter(self._dataset)
