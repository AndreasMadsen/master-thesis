
from typing import TypeVar, Iterator, Tuple
import math
import abc

import numpy as np

import tensorflow as tf

DecodeType = TypeVar('DecodeType')


class _SequenceTable:
    data: tf.TensorArray
    infered_shape: tf.TensorShape = None

    def __init__(self, size, dtype):
        # A TensorArray is required as the sequences don't have the same
        # length. Alternatively a FIFO query can be used.
        # Because the data is read more than once by the queue,
        # clear_after_read is set to False (but I can't confirm an effect).
        # Because the items has diffrent sequence lengths the infer_shape
        # is set to False. The shape is then restored in the .read method.
        self.data = tf.TensorArray(size=size,
                                   dtype=dtype,
                                   dynamic_size=False,
                                   clear_after_read=False,
                                   infer_shape=False)

    def write(self, index: int, datum: np.ndarray) -> None:
        self.data = self.data.write(index, datum)
        if self.infered_shape is None:
            self.infered_shape = tf.TensorShape((None, ) + datum.shape[1:])

    def read(self, index: tf.Tensor) -> tf.Tensor:
        datum = self.data.read(index)
        datum.set_shape(self.infered_shape)
        return datum


def _shuffle_queue(input_queue, dequeue_many=32, **kwargs):
    dequeue_op = input_queue.dequeue_many(dequeue_many)
    dtypes = [dequeue_op.dtype]
    shapes = [dequeue_op.get_shape()[1:]]

    shuffle_queue = tf.RandomShuffleQueue(
        dtypes=dtypes, shapes=shapes,
        **kwargs)
    shuffle_enqueue = shuffle_queue.enqueue_many([dequeue_op])
    tf.train.add_queue_runner(
        tf.train.QueueRunner(shuffle_queue, [shuffle_enqueue])
    )
    return shuffle_queue


class Dataset:
    source: tf.Tensor
    target: tf.Tensor
    num_observation: int
    num_batch: int

    def __init__(self,
                 observations: int,
                 dtype: np.unsignedinteger,
                 batch_size: int=32,
                 name: str='unamed',
                 shuffle=True, seed: int=None,
                 repeat=True) -> None:

        source_data = _SequenceTable(observations, dtype)
        target_data = _SequenceTable(observations, dtype)
        length_data = np.empty(observations, dtype=np.int32)

        # bucket boundaries
        # TODO: implement some naive histogram, for detecting too small buckets
        global_min_length = float('inf')
        global_max_length = 0

        for i, (source, target) in enumerate(self):
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

            # save data
            source_data.write(i, source)
            target_data.write(i, target)
            length_data[i] = max_length
            global_max_length = max(global_max_length, max_length)
            global_min_length = min(global_min_length,
                                    len(source), len(target))

        # bucket_by_sequence_length requires the input_length and tensors
        # arguments to be queues. Use a range_input_producer queue to shuffle
        # an index for sliceing the input_length and tensors laters.
        # This strategy is idendical to the one used in slice_input_producer.
        index_queue = tf.train.range_input_producer(
            len(length_data),
            name=f'dataset/{name}',
            shuffle=shuffle, seed=seed,
            num_epochs=None if repeat else 1
        )

        # To get a continues shuffling behaviour similar to suffle_batch
        # put in a RandomShuffleQueue
        if shuffle:
            data_index = _shuffle_queue(
                index_queue,
                capacity=batch_size * 128,
                min_after_dequeue=batch_size * 64,
                dequeue_many=batch_size * 32
            ).dequeue()
        else:
            data_index = index_queue.dequeue()

        # create bucket boundaries
        bucket_boundaries = [
            (i + 1) * 20 for i in range(global_max_length // 20)
        ]
        if len(bucket_boundaries) == 0:
            bucket_boundaries = [(global_min_length + global_max_length) // 2]

        # the first argument is the sequence length specifed in the
        # input_length I did not find a use for it.
        if shuffle:
            _, batch_queue = tf.contrib.training.bucket_by_sequence_length(
                input_length=tf.gather(length_data, data_index),
                tensors=[
                    source_data.read(data_index),
                    target_data.read(data_index)
                ],
                bucket_boundaries=bucket_boundaries,
                batch_size=batch_size,
                dynamic_pad=True,
                name=f'dataset/{name}',
                num_threads=32,
                capacity=batch_size * 64,
                allow_smaller_final_batch=not repeat
            )
        else:
            batch_queue = tf.train.batch(
                tensors=[
                    source_data.read(data_index),
                    target_data.read(data_index)
                ],
                batch_size=batch_size,
                dynamic_pad=True,
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

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[DecodeType, DecodeType]]:
        pass

    @abc.abstractmethod
    def encode_as_array(self, decoded: DecodeType) -> np.ndarray:
        pass
