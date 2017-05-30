
import numpy as np
import tensorflow as tf
import os.path as path

from code.metric.abstract import Metric
from code.dataset.abstract import Dataset


class ObservationLength(Metric):
    def __init__(self, dataset: Dataset, name: str="observation-length"):
        super().__init__(dataset, name=name)

    def _build_metric(self, model: 'code.model.abstract.Model') -> tf.Tensor:
        self._save_dir = path.join(model.get_save_dir(),
                                   'observation_length.csv')

        with tf.name_scope(None, self.metric_name,
                           values=[self.dataset.source,
                                   self.dataset.target, self.dataset.length]):
            lenghth_sum = tf.py_func(self._log_and_compute_sum,
                                     [self.dataset.length],
                                     self.dataset.length.dtype,
                                     stateful=False,
                                     name='observation-length-logger')

            lenghth_sum_acc = tf.get_variable(
                                name='length_sum',
                                shape=[1],
                                dtype=self.dataset.length.dtype,
                                initializer=tf.constant_initializer(0),
                                trainable=False)

            tf.add_to_collection(
                tf.GraphKeys.UPDATE_OPS,
                lenghth_sum_acc.assign(lenghth_sum_acc + lenghth_sum)
            )

            return lenghth_sum_acc

    def _log_and_compute_sum(self, lengths):
        with open(self._save_dir, "a", encoding="utf-8") as logfile:
            for length in lengths:
                print(length, file=logfile)

        return np.sum(lengths).astype(lengths.dtype)
