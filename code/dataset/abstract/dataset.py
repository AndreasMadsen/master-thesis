
from typing import TypeVar, Iterator, List, Tuple
import os
import os.path as path
import math
import abc
import tempfile
import itertools
import multiprocessing
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import bucket_by_sequence_length
from tqdm import tqdm as tqdm_bar

from code.dataset.util.length_histogram import LengthHistogram
from code.dataset.util.sequence_queue import \
    SequenceQueue, SequenceQueueMemory, SequenceQueueExternal

DecodeType = TypeVar('DecodeType')


class Dataset:
    source: tf.Tensor
    target: tf.Tensor
    num_observation: int
    num_batch: int
    batch_size: int
    data_file: str
    queue: SequenceQueue

    def __init__(self,
                 histogram: LengthHistogram,
                 dtype: np.unsignedinteger,
                 batch_size: int=32,
                 name: str='unamed',
                 shuffle: bool=True, seed: int=None,
                 repeat: bool=True,
                 external_encoding: str=None,
                 tqdm: bool=True) -> None:

        with tf.device('/cpu:0'):

            if external_encoding is not None:
                self.queue = SequenceQueueExternal(
                    external_encoding,
                    observations=histogram.observations,
                    dtype=dtype,
                    batch_size=batch_size,
                    name=f'dataset/{name}',
                    shuffle=shuffle, seed=seed,
                    repeat=repeat
                )
            else:
                self.queue = SequenceQueueMemory(
                    observations=histogram.observations,
                    dtype=dtype,
                    batch_size=batch_size,
                    name=f'dataset/{name}',
                    shuffle=shuffle, seed=seed,
                    repeat=repeat
                )

            # bucket boundaries
            if self.queue.need_data:
                for source, target in tqdm_bar(self,
                                               total=histogram.observations,
                                               unit='obs', desc='encoding',
                                               disable=not tqdm):
                    self.queue.write(*self._encode_pair(source, target))

            # dequeue dataset
            length, source, target = self.queue.read()

            if shuffle:
                # create bucket boundaries by partitioning the histogram
                bucket_boundaries = histogram.extend(1).partition(
                    min_size=batch_size * 2,
                    min_width=10
                )

                # if there are not enogth observation or spread to partition
                # the dataset, just use a batch queue.
                if len(bucket_boundaries) == 0:
                    batch = tf.train.batch(
                        tensors=[length, source, target],
                        batch_size=batch_size,
                        dynamic_pad=True,
                        name=f'dataset/{name}',
                        num_threads=32,
                        capacity=batch_size * 64,
                        allow_smaller_final_batch=not repeat
                    )
                    batch_length, batch_queue = batch[0], batch[1:]
                else:
                    # the first argument is the sequence length specifed in the
                    # input_length I did not find a use for it.
                    batch_length, batch_queue = bucket_by_sequence_length(
                        input_length=length,
                        tensors=[source, target],
                        bucket_boundaries=bucket_boundaries,
                        batch_size=batch_size,
                        dynamic_pad=True,
                        name=f'dataset/{name}',
                        num_threads=32,
                        capacity=batch_size * 64,
                        allow_smaller_final_batch=not repeat
                    )
            else:
                batch = tf.train.batch(
                    tensors=[length, source, target],
                    batch_size=batch_size,
                    dynamic_pad=True,
                    name=f'dataset/{name}',
                    num_threads=1,
                    capacity=batch_size,
                    allow_smaller_final_batch=not repeat
                )
                batch_length, batch_queue = batch[0], batch[1:]

            # make data visible
            self.length = batch_length
            self.source, self.target = batch_queue

            # calculate number of batches
            self.num_observation = histogram.observations
            self.num_batch = math.ceil(histogram.observations / batch_size)
            self.batch_size = batch_size

    def _encode_pair(self, source, target):
        # encode data
        source = self.encode_as_array(source)
        target = self.encode_as_array(target)

        # pad data
        max_length = max(len(source), len(target))
        if max_length - len(source) > 0:
            source = np.pad(source,
                            (0, max_length - len(source)), 'constant')
        if max_length - len(target) > 0:
            target = np.pad(target,
                            (0, max_length - len(target)), 'constant')

        return (max_length, source, target)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[DecodeType, DecodeType]]:
        pass

    @abc.abstractmethod
    def encode_as_array(self, decoded: DecodeType) -> np.ndarray:
        pass

    def batch_iterator(self) -> Iterator[List[Tuple[str, str]]]:
        args = [iter(self)] * self.batch_size
        batches = itertools.zip_longest(*args, fillvalue=None)

        for batch in batches:
            # remove filled elements
            batch = list(batch)
            if None in batch:
                batch = batch[:batch.index(None)]

            yield batch
