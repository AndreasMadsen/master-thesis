
import tensorflow as tf
import sugartensor as stf


@stf.sg_layer_func
def seq_dense(tensor, opt):
    r"""Applies dimentional reduction on a sequental tensor

    Args:
      tensor: A 3-D `Tensor` (automatically passed by decorator).
      opt:
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        bias: Boolean. If True, biases are added.

    Returns:
      A `Tensor` with the same type as `tensor`.
    """
    # parameter tf.sg_initializer
    w = stf.sg_initializer.he_uniform('W', (1, opt.in_dim, opt.dim))
    b = stf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    # transform parameters
    w = tf.squeeze(w, axis=0)

    # apply convolution
    out = tf.matmul(tensor, w) + b

    return out
