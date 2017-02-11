
import tensorflow as tf
import sugartensor as stf


def cross_entropy_indirect(batch_loss, name):
    stf.sg_summary_loss(batch_loss, name=f'cross_entropy/{name}')
    loss = tf.reduce_mean(batch_loss, axis=0)
    loss = tf.check_numerics(loss, f'check/cross_entropy/{name}')

    return loss
