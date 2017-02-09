
import tensorflow as tf
import sugartensor as stf  # noqa: F401


def parallel_decoder_residual_block(tensor,
                                    size=3, rate=1,
                                    name=None, reuse=False):
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
        aconv = pre_aconv.sg_aconv1d(causal=True,
                                     size=size, rate=rate,
                                     act='relu', ln=True,
                                     name="conv-dilated")

        # dimension recover and residual connection
        out = aconv.sg_conv1d(size=1, dim=in_dim,
                              name="recover-dim") + tensor

        return out
