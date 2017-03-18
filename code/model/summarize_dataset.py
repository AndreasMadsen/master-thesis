
import math
from typing import Optional

from tqdm import tqdm
import sugartensor as stf

from code.model.abstract.model import Model
from code.dataset.abstract.text_dataset import TextDataset


class SummarizeDataset(Model):
    observations: int = 0
    max_length: int = 0
    min_length: int = 1e9

    def __init__(self, dataset: TextDataset) -> None:
        super().__init__(dataset)

    def train(self) -> None:
        for source, target in tqdm(self.dataset,
                                   unit='obs', desc='summarize',
                                   total=self.dataset.num_observation):
            self.observations += 1

            length = max(len(source), len(target)) + 1

            update = False
            if length > self.max_length:
                self.max_length = length
                update = True
            if length < self.min_length:
                self.min_length = length
                update = True
            if update:
                self.print_stats()

    def print_stats(self):
        tqdm.write(f'{self.observations} obs: min={self.min_length}, ' +
                   f'max={self.max_length}')
