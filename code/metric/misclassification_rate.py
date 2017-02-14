
import tensorflow as tf

from code.metric.abstract import Metric
from code.dataset.abstract import Dataset


class MisclassificationRate(Metric):
    def __init__(self, dataset: Dataset, name: str="misclassification-rate"):
        super().__init__(dataset, name=name)

    def _build_metric(self, model: 'code.model.abstract.Model') -> tf.Tensor:
        with tf.name_scope(None, self.metric_name,
                           values=[self.dataset.source, self.dataset.target]):
            x = self.dataset.source
            y = self.dataset.target

            # build mask
            mask = tf.cast(tf.not_equal(y, tf.zeros_like(y)), tf.float32)

            # create masked error tensor
            errors = tf.not_equal(model.inference_model(x, reuse=True), y)
            errors = tf.cast(errors, tf.float32) * mask  # mask errors

            # tf.sum(mask) is the number of unmasked elements
            return tf.reduce_sum(errors) / tf.reduce_sum(mask)
