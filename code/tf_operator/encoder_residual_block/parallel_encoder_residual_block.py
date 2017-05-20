
import sugartensor as stf  # noqa: F401
import tensorflow as tf

from code.tf_operator.convolution import aconv1d


def parallel_encoder_residual_block(tensor,
                                    size=3, rate=1,
                                    normalization='bn',
                                    name=None, reuse=None):
    default_name = f"encoder-res-block-{size}-{rate}"

    # use layer normalization for ByteNet v2
    normalize = {
        'bn': normalization == 'bn',
        'ln': normalization == 'ln'
    }

    with tf.variable_scope(name, default_name, [tensor], reuse=reuse):
        # input dimension
        in_dim = tensor.get_shape().as_list()[-1]

        # reduce dimension
        pre_aconv = tensor.sg_bypass(act='relu', **normalize, scale=False,
                                     name="activation")
        pre_aconv = pre_aconv.sg_conv1d(size=1, dim=in_dim // 2,
                                        act='relu', **normalize, scale=False,
                                        name="reduce-dim")

        # 1xk conv dilated
        aconv = aconv1d(pre_aconv,
                        size=size, rate=rate,
                        act='relu', **normalize, scale=False,
                        name="conv-dilated")

        # dimension recover and residual connection
        out = aconv.sg_conv1d(size=1, dim=in_dim,
                              name="recover-dim") + tensor

        return out
