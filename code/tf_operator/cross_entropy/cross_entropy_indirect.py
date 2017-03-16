
import tensorflow as tf
import sugartensor as stf


def cross_entropy_indirect(logprops, name):
    loss = tf.reduce_mean(-logprops, axis=0)
    loss = tf.check_numerics(loss, f'check/cross_entropy/{name}')

    return loss
