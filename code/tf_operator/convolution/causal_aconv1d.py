
import sugartensor as stf
import tensorflow as tf

from code.tf_operator.convolution.seq_causal_aconv1d import \
    seq_causal_aconv1d


def _seq_aconv1d_init(tensor,
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


@stf.sg_layer_func
def _fast_causal_aconv1d(tensor, opt):
    # default options
    opt += stf.sg_opt(size=2, rate=1, pad='SAME')

    # parameter tf.sg_initializer
    w = stf.sg_initializer.he_uniform('W', (opt.size, opt.in_dim, opt.dim))
    b = stf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    pad_len = (opt.size - 1) * opt.rate  # padding size
    x = tf.pad(tensor, [[0, 0], [pad_len, 0], [0, 0]])

    out = tf.nn.convolution(input=x, filter=w,
                            padding='VALID',
                            dilation_rate=[opt.rate]) + b

    return out


def causal_aconv1d(tensor, low_memory=False, size=2, rate=1, **kwargs):
    if low_memory:
        def scan_op(acc, tensor_t):
            (state_tm1, aconv_tm1) = acc

            aconv_t = seq_causal_aconv1d(tensor_t, previous=state_tm1,
                                         size=size, rate=rate,
                                         **kwargs)

            return ((tensor_t, ) + state_tm1[:-1], aconv_t)

        (_, aconv) = tf.scan(
            scan_op,
            elems=tf.transpose(tensor, perm=[1, 0, 2]),
            initializer=(
                _seq_aconv1d_init(tensor, size=size, rate=rate),
                tf.zeros(
                    (tf.shape(tensor)[0], tensor.get_shape().as_list()[-1]),
                    dtype=tensor.dtype
                ),  # aconv
            )
        )

        aconv = tf.transpose(aconv, perm=[1, 0, 2])
        aconv.set_shape(tensor.get_shape())
    else:
        aconv = _fast_causal_aconv1d(tensor,
                                     size=size, rate=rate,
                                     **kwargs)

    return aconv
