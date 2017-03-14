
import numpy as np
import tensorflow as tf


class SequenceTable:
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
