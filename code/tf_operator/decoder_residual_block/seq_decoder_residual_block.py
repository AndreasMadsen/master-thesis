
import tensorflow as tf
import sugartensor as stf  # noqa: F401

from code.tf_operator.seq_dense import seq_dense
from code.tf_operator.convolution import seq_causal_aconv1d


def seq_decoder_residual_block_init(tensor,
                                    in_dim=None,
                                    block_type='bytenet',
                                    size=3, rate=1):
    default_name = f"seq-decoder-res-block-{size}-{rate}-init"

    with tf.name_scope(None, default_name, [tensor]):
        # input dimension
        batches = tf.shape(tensor)[0]
        if in_dim is None:
            in_dim = tensor.get_shape().as_list()[-1]

        if block_type == 'bytenet':
            aconv_dim = in_dim // 2
        elif block_type == 'small':
            aconv_dim = in_dim
        else:
            raise NotImplementedError(
                f'Block type {block_type} is not implemented'
            )

        # create zero array
        previous_size = (size - 1) * rate
        init = tf.zeros(
            (batches, aconv_dim),
            dtype=tensor.dtype,
            name="seq-decoder-residual-block-init-zero"
        )

        # repeat zero
        return (init, ) * previous_size


def seq_decoder_residual_block(tensor, previous,
                               size=3, rate=1,
                               act='relu',
                               normalization='ln',
                               block_type='bytenet',
                               name=None, reuse=None):
    default_name = f"seq-decoder-res-block-{size}-{rate}"

    # use check normalization
    normalize = {
        'ln': normalization == 'ln'
    }

    scope_variables = [tensor] + list(previous)
    with tf.variable_scope(name, default_name, scope_variables, reuse=reuse):
        # input dimension
        in_dim = tensor.get_shape().as_list()[-1]

        # reduce dimension
        if block_type == 'bytenet':
            pre_aconv = tensor.sg_bypass(act=act, **normalize, scale=False,
                                         name="activation")
            pre_aconv = seq_dense(pre_aconv, dim=in_dim // 2,
                                  act=act, **normalize, scale=False,
                                  name="reduce-dim")

            # 1xk conv dilated
            aconv = seq_causal_aconv1d(pre_aconv, previous=previous,
                                       size=size, rate=rate,
                                       act=act, **normalize, scale=False,
                                       name="conv-dilated")

            # dimension recover and residual connection
            out = seq_dense(aconv,
                            dim=in_dim,
                            name="recover-dim") + tensor
        elif block_type == 'small':
            pre_aconv = tensor.sg_bypass(act=act, **normalize, scale=False,
                                         name="activation")

            # 1xk conv dilated
            aconv = seq_causal_aconv1d(pre_aconv, previous=previous,
                                       size=size, rate=rate,
                                       name="conv-dilated")
            # residual connection
            out = aconv + tensor
        else:
            raise NotImplementedError(
                f'Block type {block_type} is not implemented'
            )

        # return (
        #  the input for the same layer in next iteration
        #  the input for the next layer in same iteration
        # )
        return ((pre_aconv, ) + previous[:-1], out)
