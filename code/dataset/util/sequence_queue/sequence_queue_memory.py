
from typing import Tuple

import tensorflow as tf
import numpy as np

from code.dataset.util.sequence_queue.sequence_queue import SequenceQueue
from code.tf_operator import \
    shuffle_tensor_index, \
    SequenceTable


class SequenceQueueMemory(SequenceQueue):
    source_data: SequenceTable
    target_data: SequenceTable
    length_data: np.ndarray
    incremented_index: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(True, *args, **kwargs)

        self.source_data = SequenceTable(self.observations, self.dtype)
        self.target_data = SequenceTable(self.observations, self.dtype)
        self.length_data = np.empty(self.observations, dtype=np.int32)

    def write(self,
              length: int, source: np.ndarray, target: np.ndarray) -> None:
        self.source_data.write(self.incremented_index, source)
        self.target_data.write(self.incremented_index, target)
        self.length_data[self.incremented_index] = length
        self.incremented_index += 1

    def read(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # bucket_by_sequence_length requires the input_length and tensors
        # arguments to be queues. Use a range_input_producer queue to shuffle
        # an index for sliceing the input_length and tensors laters.
        # This strategy is idendical to the one used in slice_input_producer.
        data_index = tf.train.range_input_producer(
            self.observations,
            name=f'dataset/{self.name}',
            shuffle=self.shuffle, seed=self.seed,
            num_epochs=None if self.repeat else 1
        )

        # To get a continues shuffling behaviour similar to suffle_batch
        # put in a RandomShuffleQueue
        if self.shuffle:
            data_index = shuffle_tensor_index(
                data_index,
                capacity=self.batch_size * 128,
                min_after_dequeue=self.batch_size * 64,
                dequeue_many=self.batch_size * 32,
                seed=self.seed
            )
        else:
            data_index = data_index.dequeue()

        return (
            tf.gather(self.length_data, data_index),
            self.source_data.read(data_index),
            self.target_data.read(data_index)
        )
