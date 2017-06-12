
import tensorflow as tf
import sugartensor as stf  # noqa: F401

from code.tf_operator.convolution import causal_aconv1d


def parallel_decoder_residual_block(tensor,
                                    size=3, rate=1,
                                    low_memory=False,
                                    act='relu',
                                    normalization='ln',
                                    block_type='bytenet',
                                    name=None, reuse=None):
    default_name = f"decoder-res-block-{size}-{rate}"

    # use check normalization
    normalize = {
        'ln': normalization == 'ln'
    }

    with tf.variable_scope(name, default_name, [tensor], reuse=reuse):
        # input dimension
        in_dim = tensor.get_shape().as_list()[-1]

        if block_type == 'bytenet':
            # reduce dimension
            pre_aconv = tensor.sg_bypass(act=act, **normalize, scale=False,
                                         name="activation")
            pre_aconv = pre_aconv.sg_conv1d(size=1, dim=in_dim // 2,
                                            act=act, **normalize, scale=False,
                                            name="reduce-dim")

            # 1xk conv dilated
            aconv = causal_aconv1d(pre_aconv, size=size, rate=rate,
                                   act=act, **normalize, scale=False,
                                   low_memory=low_memory,
                                   name="conv-dilated")

            # dimension recover and residual connection
            out = aconv.sg_conv1d(size=1, dim=in_dim,
                                  name="recover-dim") + tensor
        elif block_type == 'small':
            # activate and normalize input
            pre_aconv = tensor.sg_bypass(act=act, **normalize, scale=False,
                                         name="activation")
            # 1xk conv dilated
            aconv = causal_aconv1d(pre_aconv, size=size, rate=rate,
                                   low_memory=low_memory,
                                   name="conv-dilated")

            # residual connection
            out = aconv + tensor
        else:
            raise NotImplementedError(
                f'Block type {block_type} is not implemented'
            )

        return out
