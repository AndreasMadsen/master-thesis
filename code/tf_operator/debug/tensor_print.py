
import numpy as np
import tensorflow as tf


def tensor_print(tensor, message, threshold=100):
    def numpy_printer(ndarray):
        print(f'> {message}:')

        old_threshold = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=threshold)
        print(ndarray)
        np.set_printoptions(threshold=old_threshold)

        return ndarray

    out = tf.py_func(numpy_printer, [tensor], tensor.dtype)
    out.set_shape(tensor.get_shape())
    return out
