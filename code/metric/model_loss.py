
import tensorflow as tf

from code.metric.abstract import Metric
from code.dataset.abstract import Dataset


class ModelLoss(Metric):
    def __init__(self, dataset: Dataset, name: str="model-loss"):
        super().__init__(dataset, name=name)

    def _build_metric(self, model: 'code.model.abstract.Model') -> tf.Tensor:
        with tf.name_scope(None, self.metric_name,
                           values=[self.dataset.source, self.dataset.target]):
            loss, losses = model.loss_model(
                self.dataset.source, self.dataset.target,
                reuse=True
            )
            return loss
