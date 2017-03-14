
from typing import Tuple
import os
import os.path as path

import tensorflow as tf
import numpy as np

from code.dataset.util.sequence_queue.sequence_queue import SequenceQueue
from code.tf_operator import \
    shuffle_tensor_list, \
    make_sequence_example, parse_sequence_example


class SequenceQueueExternal(SequenceQueue):
    data_file: str
    writer = tf.python_io.TFRecordWriter

    def __init__(self, external_encoding: str, *args, **kwargs):
        self.data_file = path.realpath(external_encoding)

        # detect if data file exists
        has_data = (path.exists(self.data_file) and
                    os.stat(self.data_file).st_size > 0)
        super().__init__(not has_data, *args, **kwargs)

        if self.need_data:
            os.makedirs(path.dirname(self.data_file), exist_ok=True)
            with open(self.data_file, 'w'):
                self.writer = tf.python_io.TFRecordWriter(
                    self.data_file,
                    options=tf.python_io.TFRecordOptions(
                        tf.python_io.TFRecordCompressionType.ZLIB
                    )
                )

    def write(self,
              length: int, source: np.ndarray, target: np.ndarray) -> None:
        if not self.need_data:
            raise RuntimeError(
                'queue.write should not be called when need_data is false'
            )

        example = make_sequence_example(length, source, target)
        self.writer.write(example.SerializeToString())

    def read(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if self.need_data:
            self.writer.close()

        # create filename queue of one filename. TFRecordReader demands this.
        filename_queue = tf.train.string_input_producer(
            [self.data_file],
            num_epochs=None if self.repeat else 1
        )

        # read serialized data
        reader = tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(
                tf.python_io.TFRecordCompressionType.ZLIB
            )
        )
        reader_dequeue = reader.read(filename_queue)

        # parse data
        length, source, target = parse_sequence_example(reader_dequeue.value)

        # cast to original type
        length = tf.cast(length, dtype=tf.int32)
        source = tf.cast(source, dtype=self.dtype)
        target = tf.cast(target, dtype=self.dtype)

        # To get a continues shuffling behaviour similar to suffle_batch
        # put in a RandomShuffleQueue
        if self.shuffle:
            length, source, target = shuffle_tensor_list(
                (length, source, target),
                capacity=self.batch_size * 128,
                min_after_dequeue=self.batch_size * 64,
                seed=self.seed
            )

        return (length, source, target)
