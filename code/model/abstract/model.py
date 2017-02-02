
import abc

from code.dataset import Dataset


class Model:
    dataset: Dataset

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    @abc.abstractmethod
    def train(self) -> None:
        pass

    @abc.abstractmethod
    def predict(self) -> None:
        pass
