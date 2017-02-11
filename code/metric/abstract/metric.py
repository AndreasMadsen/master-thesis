
import abc

import tensorflow as tf

from code.dataset.abstract import Dataset


class Metric:
    dataset: Dataset
    metric_name: str

    def __init__(self, dataset: Dataset, name: str) -> None:
        self.dataset = dataset
        self.metric_name = name

    @abc.abstractmethod
    def _build_metric(self, dataset: Dataset) -> tf.Tensor:
        pass

    def build(self, model: 'code.model.abstract.Model') -> tf.Tensor:
        metric = self._build_metric(model)
        return tf.identity(metric, name=self.metric_name)
