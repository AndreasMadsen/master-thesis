
import sugartensor as stf
import tensorflow as tf


def optimizer(optim='MaxProp',
              lr=0.001, beta1=0.9, beta2=0.99, momentum=0.):
    # select optimizer
    if optim == 'MaxProp':
        opt = stf.sg_optimize.MaxPropOptimizer(
            learning_rate=lr, beta2=beta2
        )
    elif optim == 'AdaMax':
        opt = stf.sg_optimize.AdaMaxOptimizer(
            learning_rate=lr, beta1=beta1, beta2=beta2
        )
    elif optim == 'Adam':
        opt = tf.train.AdamOptimizer(
            learning_rate=lr, beta1=beta1, beta2=beta2
        )
    elif optim == 'RMSProp':
        opt = tf.train.RMSPropOptimizer(
            learning_rate=lr, decay=beta1, momentum=momentum
        )
    else:
        opt = tf.train.GradientDescentOptimizer(
            learning_rate=lr
        )

    return opt
