
import tensorflow as tf
import sugartensor as stf


def cross_entropy_direct(logits, target, name):
    loss = logits.sg_ce(target=target, mask=True, name=f'cross_entropy/{name}')
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)
    loss = tf.check_numerics(loss, f'check/cross_entropy/{name}')

    return loss
