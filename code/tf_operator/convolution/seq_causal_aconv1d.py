
import tensorflow as tf
import sugartensor as stf


@stf.sg_layer_func
def seq_causal_aconv1d(tensor, opt):
    r"""Applies 1-D atrous (or dilated) convolution.

    Args:
      tensor: A 3-D `Tensor` (automatically passed by decorator).
      opt:
        size: A positive `integer` representing `[kernel width]`. As a default
          it is set to 2 if causal is True, 3 otherwise.
        rate: A positive `integer`. The stride with which we sample input
          values across the `height` and `width` dimensions. Default is 1.
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        bias: Boolean. If True, biases are added.

    Returns:
      A `Tensor` with the same type as `tensor`.
    """
    # default options
    opt += stf.sg_opt(previous=tuple(), size=2, rate=1, summary=True)

    # check length of previous
    if (len(opt.previous) != (opt.size - 1) * opt.rate):
        raise LookupError(
            f'previous should have length {(opt.size - 1) * opt.rate}' +
            f' but had {len(opt.previous)}.'
        )

    # get shapes
    batches = tf.shape(tensor)[0]

    # parameter tf.sg_initializer
    w = stf.sg_initializer.he_uniform('W', (opt.size, opt.in_dim, opt.dim),
                                      summary=opt.summary)
    if opt.bias:
        b = stf.sg_initializer.constant('b', opt.dim, summary=opt.summary)
    else:
        b = 0

    # construct "image" for convolution
    image = [
        opt.previous[size_i * opt.rate - 1]
        for size_i in range(opt.size - 1, 0, -1)
    ] + [tensor]
    image = tf.stack(image, axis=1)  # [batches, opt.size, opt.in_dim]

    # apply convolution
    image = tf.reshape(image, (batches, opt.size * opt.in_dim))
    w = tf.reshape(w, (opt.size * opt.in_dim, opt.dim))
    out = tf.matmul(image, w) + b

    return out
