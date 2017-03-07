
import sugartensor as stf
import tensorflow as tf

from code.tf_operator.parallel.tower_gradient import tower_gradient


def _get_optimizer_instance(optim='MaxProp',
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


def _category_filter(ops, category=''):
    if isinstance(category, (tuple, list)):
        filtered = []
        for cat in category:
            filtered.extend([t for t in ops if t.name.startswith(cat)])
    else:
        filtered = [t for t in ops if t.name.startswith(category)]

    return filtered


def tower_optim(losses, category='', **kwargs):
    opt = _get_optimizer_instance(**kwargs)

    # get trainable variables
    var_list = _category_filter(tf.trainable_variables(), category)

    # calc gradients (represented as (grad, var) pairs) like compute_gradients
    gradient = tower_gradient(opt, losses, var_list)

    # add summary
    for v, g in zip(var_list, gradient):
        # exclude batch normal statics
        if 'mean' not in v.name and 'variance' not in v.name \
                and 'beta' not in v.name and 'gamma' not in v.name:
            stf.sg_summary_gradient(v, g)

    # gradient update op
    with tf.device('/cpu:0'):
        grad_op = opt.apply_gradients(
            gradient, global_step=stf.sg_global_step()
        )

    # extra update ops within category
    update_op = _category_filter(
        tf.get_collection(tf.GraphKeys.UPDATE_OPS), category
    )
    # Because the tower adds multiple UPDATE_OPS for each batch normalization
    # variable (beta and gamma), remove the dublicate update ops.
    # This will only mean over a partial minibatch, but since the moving
    # avegers conveges very quickly this does not have a pratical effect.
    unique_update_op = dict()
    for tensor in update_op:
        if tensor.op.type != 'Assign':
            raise TypeError('tensor {tensor} is not an assign op')

        # get the name of the variable that the assign op will change
        unique_update_op[tensor.op.inputs[0].name] = tensor
    update_op = list(unique_update_op.values())

    return [grad_op] + update_op
