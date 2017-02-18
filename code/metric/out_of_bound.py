
import tensorflow as tf

from code.metric.abstract import Metric
from code.dataset.abstract import Dataset


class OutOfBound(Metric):
    def __init__(self, dataset: Dataset, name: str="out-of-bound"):
        super().__init__(dataset, name=name)

    def _build_metric(self, model: 'code.model.abstract.Model') -> tf.Tensor:
        with tf.name_scope(None, self.metric_name,
                           values=[self.dataset.source]):
            x = self.dataset.source

            # create masked error tensor
            predict = model.inference_model(x, reuse=True)
            out_of_bound = tf.greater_equal(
                predict,
                tf.cast(model.dataset.vocabulary_size, dtype=predict.dtype)
            )

            return tf.reduce_sum(tf.cast(out_of_bound, tf.int32))
