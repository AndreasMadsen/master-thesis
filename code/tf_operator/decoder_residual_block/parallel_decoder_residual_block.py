
import tensorflow as tf
import sugartensor as stf  # noqa: F401

from code.tf_operator.decoder_residual_block.seq_causal_aconv1d \
    import seq_causal_aconv1d


def seq_aconv1d_init(tensor,
                     size=3, rate=1):
    default_name = f"seq-aconv1d-{size}-{rate}-init"

    with tf.name_scope(None, default_name, [tensor]):
        # input dimension
        batches = tf.shape(tensor)[0]
        in_dim = tensor.get_shape().as_list()[-1]

        # create zero array
        previous_size = (size - 1) * rate
        init = tf.zeros(
            (batches, in_dim),
            dtype=tensor.dtype,
            name="seq-aconv1d-init-zero"
        )

        # repeat zero
        return (init, ) * previous_size


def parallel_decoder_residual_block(tensor,
                                    size=3, rate=1,
                                    low_memory=False,
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
        if low_memory:
            def scan_op(acc, pre_aconv_t):
                (state_tm1, aconv_tm1) = acc

                aconv_t = seq_causal_aconv1d(pre_aconv_t, previous=state_tm1,
                                             size=size, rate=rate,
                                             act='relu', ln=True,
                                             name="conv-dilated")

                return ((pre_aconv_t, ) + state_tm1[:-1], aconv_t)

            (_, aconv) = tf.scan(
                scan_op,
                elems=tf.transpose(pre_aconv, perm=[1, 0, 2]),
                initializer=(
                    seq_aconv1d_init(pre_aconv, size=size, rate=rate),
                    tf.zeros(
                        (tf.shape(pre_aconv)[0], in_dim // 2),
                        dtype=pre_aconv.dtype
                    ),  # aconv
                )
            )

            aconv = tf.transpose(aconv, perm=[1, 0, 2])
            aconv.set_shape(pre_aconv.get_shape())
        else:
            aconv = pre_aconv.sg_aconv1d(causal=True,
                                         size=size, rate=rate,
                                         act='relu', ln=True,
                                         name="conv-dilated")

        # dimension recover and residual connection
        out = aconv.sg_conv1d(size=1, dim=in_dim,
                              name="recover-dim") + tensor

        return out
