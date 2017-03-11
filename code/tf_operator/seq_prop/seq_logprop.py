
import tensorflow as tf


def seq_logprop(x, mask=None, axis=1, name=None):
    with tf.name_scope(name, "seq-logprop", values=[x, mask]):

        if mask is not None:
            # elementwise: x if mask != 0 else 1
            condition = tf.not_equal(mask, 0)
            x = tf.where(condition, x=x, y=tf.zeros_like(x))

        t = tf.reduce_sum(x, axis=axis)

        return t
