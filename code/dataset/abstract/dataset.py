
import math

import numpy as np

import tensorflow as tf


class Dataset:
    source: tf.Tensor
    target: tf.Tensor
    num_observation: int
    num_batch: int

    def __init__(self, sources: np.ndarray, targets: np.ndarray,
                 batch_size: int=32,
                 observations: int=None,
                 name: str='unamed',
                 shuffle=True, seed: int=None,
                 repeat=True) -> None:

        # take top `observations` from sources and targets
        if observations is not None:
            sources = sources[:observations]
            targets = targets[:observations]

        # to constant tensor
        sources = tf.convert_to_tensor(sources)
        targets = tf.convert_to_tensor(targets)

        # get shape
        observations = int(sources.get_shape()[0])

        # create queue from constant tensor
        source, target = tf.train.slice_input_producer(
            [sources, targets],
            name=f'dataset/{name}',
            shuffle=shuffle, seed=seed,
            num_epochs=None if repeat else 1
        )

        # create batch queue
        if shuffle:
            batch_queue = tf.train.shuffle_batch(
                [source, target], batch_size,
                name=f'dataset/{name}',
                seed=seed,
                num_threads=32,
                capacity=batch_size * 64,
                min_after_dequeue=batch_size * 32,
                allow_smaller_final_batch=not repeat
            )
        else:
            batch_queue = tf.train.batch(
                [source, target], batch_size,
                name=f'dataset/{name}',
                num_threads=1,
                capacity=batch_size,
                allow_smaller_final_batch=not repeat
            )

        # make data visible
        self.source, self.target = batch_queue

        # calculate number of batches
        self.num_observation = observations
        self.num_batch = math.ceil(observations / batch_size)
