
import tensorflow as tf
import sugartensor as stf


def cross_entropy_summary(tensor, name, reuse=False):
    if not reuse and not tf.get_variable_scope().reuse:
        tf.summary.scalar(f'losses/cross_entropy/{name}', tensor)

    return tensor
