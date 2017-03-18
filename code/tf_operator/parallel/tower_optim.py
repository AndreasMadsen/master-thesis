
import sugartensor as stf
import tensorflow as tf

from code.tf_operator.parallel.tower_gradient import tower_gradient
from code.tf_operator.train.optimizer import optimizer


def tower_optim(losses, **kwargs):
    opt = optimizer(**kwargs)

    # get trainable variables
    var_list = tf.trainable_variables()

    # calc gradients (represented as (grad, var) pairs) like compute_gradients
    gradient = tower_gradient(opt, losses, var_list)

    # add summary
    for g, v in gradient:
        # exclude batch normal statics
        if 'mean' not in v.name and 'variance' not in v.name \
                and 'beta' not in v.name and 'gamma' not in v.name:
            stf.sg_summary_gradient(v, gradient=g)

    # gradient update op
    with tf.device('/cpu:0'):
        grad_op = opt.apply_gradients(
            gradient, global_step=stf.sg_global_step()
        )

    # extra update ops within category
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
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
