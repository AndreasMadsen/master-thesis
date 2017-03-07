
import tensorflow as tf


def mean_n(tensor_list, name=None):
    with tf.name_scope(name, 'mean_n', values=tensor_list):
        stacked_tensor = tf.stack(tensor_list)
        return tf.reduce_mean(stacked_tensor, 0)
