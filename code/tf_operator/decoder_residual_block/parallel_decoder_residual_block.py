
import tensorflow as tf
import sugartensor as stf  # noqa: F401

from code.tf_operator.convolution import causal_aconv1d


def parallel_decoder_residual_block(tensor,
                                    size=3, rate=1,
                                    low_memory=False,
                                    name=None, reuse=None):
    default_name = f"decoder-res-block-{size}-{rate}"

    with tf.variable_scope(name, default_name, [tensor], reuse=reuse):
        # input dimension
        in_dim = tensor.get_shape().as_list()[-1]

        # reduce dimension
        pre_aconv = tensor.sg_bypass(act='relu', ln=True,
                                     name="activation")
        pre_aconv = pre_aconv.sg_conv1d(size=1, dim=in_dim // 2,
                                        act='relu', ln=True,
                                        name="reduce-dim")

        # 1xk conv dilated
        aconv = causal_aconv1d(pre_aconv, size=size, rate=rate,
                               act='relu', ln=True,
                               low_memory=low_memory,
                               name="conv-dilated")

        # dimension recover and residual connection
        out = aconv.sg_conv1d(size=1, dim=in_dim,
                              name="recover-dim") + tensor

        return out
