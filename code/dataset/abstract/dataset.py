
import math

import numpy as np

import tensorflow as tf


class Dataset:
    source: tf.Tensor
    target: tf.Tensor
    num_observation: int
    num_batch: int

    def __init__(self, sources: np.ndarray, targets: np.ndarray,
                 batch_size: int=32, name: str='train') -> None:

        # to constant tensor
        sources = tf.convert_to_tensor(sources)
        targets = tf.convert_to_tensor(targets)

        # create queue from constant tensor
        source, target = tf.train.slice_input_producer([sources, targets])

        # create batch queue
        batch_queue = tf.train.shuffle_batch([source, target], batch_size,
                                             num_threads=32,
                                             capacity=batch_size * 64,
                                             min_after_dequeue=batch_size * 32,
                                             name=name)

        # make data visible
        self.source, self.target = batch_queue

        # calculate number of batches
        self.num_observation = int(sources.get_shape()[0])
        self.num_batch = math.ceil(self.num_observation / batch_size)
