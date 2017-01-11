
import abc

from dataset.abstract.dataset import Dataset


class Model:
    dataset: Dataset

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    @abc.abstractmethod
    def train(self) -> None:
        pass
