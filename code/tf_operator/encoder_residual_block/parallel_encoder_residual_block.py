
import sugartensor as stf
import tensorflow as tf


def parallel_encoder_residual_block(tensor,
                                    size=3, rate=1,
                                    name=None, reuse=False):
    default_name = f"encoder-res-block-{size}-{rate}"

    with tf.variable_scope(name, default_name, [tensor], reuse=reuse):
        # input dimension
        in_dim = tensor.get_shape().as_list()[-1]

        # reduce dimension
        pre_aconv = tensor.sg_bypass(act='relu', bn=True,
                                     name="activation")
        pre_aconv = pre_aconv.sg_conv1d(size=1, dim=in_dim // 2,
                                        act='relu', bn=True,
                                        name="reduce-dim")

        # 1xk conv dilated
        aconv = pre_aconv.sg_aconv1d(size=size, rate=rate,
                                     act='relu', bn=True,
                                     name="conv-dilated")

        # dimension recover and residual connection
        out = aconv.sg_conv1d(size=1, dim=in_dim,
                              name="recover-dim") + tensor

        return out
