
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

    def extend(self, length):
        """add length to all sequences, causeing an offset in the histogram"""
        return LengthHistogram([0] * length + self.data)

    def partition(self, min_size=5, min_width=1):
        """return length values that corresponds to slices

        for example [2, 3, 6] => [:2, 2:3, 3:6, 6:]
        """

        splits = []
        last_split = None
        next_split = min(min_width, len(self.data))
        split_size = sum(self.data[:next_split])

        while next_split < len(self.data):
            # the split is to small, make it bigger
            if split_size < min_size:
                split_size += self.data[next_split]
                next_split += 1
            # the split is good and it and update
            else:
                splits.append(next_split)
                last_split = next_split
                next_split += min(min_width, len(self.data))
                split_size = sum(self.data[last_split:next_split])

        # make sure the last split is big enogth
        if sum(self.data[last_split:]) < min_size and len(splits) > 0:
            splits.pop()

        return splits
