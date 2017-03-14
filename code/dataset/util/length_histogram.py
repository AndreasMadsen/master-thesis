
from typing import List


class LengthHistogram:
    data: List

    def __init__(self, data: List[int]=None) -> None:
        if data is None:
            self.data = [0]
        else:
            if len(data) <= 1:
                raise ValueError('data must contain data')
            self.data = list(data)

    def add(self, length: int) -> None:
        if length >= len(self):
            for i in range(len(self), length + 1):
                self.data.append(0)

        self.data[length] += 1

    @property
    def min_length(self):
        for i, count in enumerate(self.data):
            if count != 0:
                return i

    @property
    def max_length(self):
        return len(self.data) - 1

    @property
    def observations(self):
        return sum(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> int:
        if isinstance(index, int):
            return self.data[index]

        return sum(self.data[index])

    def __repr__(self) -> str:
        return f'LengthHistogram{repr(self.data[1:])}'

    def encode(self) -> List[int]:
        return self.data

    @classmethod
    def decode(cls, data: List[int]) -> None:
        return clc(data)
