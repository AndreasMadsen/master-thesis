
import sugartensor as stf
import tensorflow as tf


@stf.sg_layer_func
def aconv1d(tensor, opt):
    # default options
    opt += stf.sg_opt(size=2, rate=1, pad='SAME', summary=True)

    # parameter tf.sg_initializer
    w = stf.sg_initializer.he_uniform('W', (opt.size, opt.in_dim, opt.dim),
                                      summary=opt.summary)
    if opt.bias:
        b = stf.sg_initializer.constant('b', opt.dim, summary=opt.summary)
    else:
        b = 0

    # apply 1d convolution
    out = tf.nn.convolution(input=tensor, filter=w,
                            padding='SAME',
                            dilation_rate=[opt.rate]) + b

    return out
