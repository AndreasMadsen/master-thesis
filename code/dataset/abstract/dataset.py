
import math

import numpy as np

import tensorflow as tf


class Dataset:
    source: tf.Tensor
    target: tf.Tensor
    num_observation: int
    num_batch: int

    def __init__(self, sources: np.ndarray, targets: np.ndarray,
                 batch_size: int=32, consistent=False,
                 name: str='train', seed: int=None) -> None:

        # to constant tensor
        sources = tf.convert_to_tensor(sources)
        targets = tf.convert_to_tensor(targets)

        # create queue from constant tensor
        source, target = tf.train.slice_input_producer([sources, targets],
                                                       seed=seed)

        suffle_args = {
            'num_threads': 32,
            'capacity': batch_size * 64,
            'min_after_dequeue': batch_size * 32,
            'allow_smaller_final_batch': False
        }
        if consistent:
            observations = int(sources.get_shape()[0])
            suffle_args['num_threads'] = 1
            suffle_args['capacity'] = observations
            suffle_args['min_after_dequeue'] = 0
            batch_size = observations

        # create batch queue
        batch_queue = tf.train.shuffle_batch([source, target], batch_size,
                                             name=name, seed=seed,
                                             **suffle_args)

        # make data visible
        self.source, self.target = batch_queue

        # calculate number of batches
        self.num_observation = int(sources.get_shape()[0])
        self.num_batch = math.ceil(self.num_observation / batch_size)
