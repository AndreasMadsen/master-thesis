
import tensorflow as tf
import sugartensor as stf


def cross_entropy_direct(logits, target, name, reuse=None):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=target)
    loss *= tf.cast(tf.not_equal(target, tf.zeros_like(target)), tf.float32)

    batch_loss = tf.reduce_sum(loss, axis=1)
    if not reuse and not tf.get_variable_scope().reuse:
        stf.sg_summary_loss(batch_loss, name=f'cross_entropy/{name}')

    batch_loss = tf.reduce_mean(batch_loss, axis=0)
    batch_loss = tf.check_numerics(batch_loss, f'check/cross_entropy/{name}')

    return batch_loss
