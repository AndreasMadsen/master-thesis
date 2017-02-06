
import tensorflow as tf


def seq_prop(x, mask=None, axis=1, name=None):
    with tf.name_scope(name, "seq-prop", values=[x, mask]):

        if mask is not None:
            # elementwise: x if mask != 0 else 1
            condition = tf.not_equal(mask, tf.zeros_like(mask))
            x = tf.where(condition, x=x, y=tf.ones_like(x))

        t = tf.reduce_prod(x, axis=axis)

        return t
