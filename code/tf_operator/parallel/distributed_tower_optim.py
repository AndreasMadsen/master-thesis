
import sugartensor as stf
import tensorflow as tf

from code.tf_operator.parallel.tower_gradient import tower_gradient
from code.tf_operator.train.optimizer import optimizer


def distributed_tower_optim(losses, **kwargs):
    '''
    losses = {
        cat0: [(gpu0, loss0), (gpu2, loss1)],
        cat1: [(gpu1, loss0), (gpu3, loss1)]
    }
    distribution = [cat0, cat1]
    '''

    opt = optimizer(**kwargs)

    gradient = []
    var_list = tf.trainable_variables()
    for category_name, category_tower_losses in losses.items():
        # get trainable variables
        category_vars = [
            var for var in var_list if var.name.startswith(category_name)
        ]

        # calc gradients (represented as (grad, var) pairs) like
        # compute_gradients
        gradient += tower_gradient(opt, category_tower_losses, category_vars)

    # add summary
    for g, v in gradient:
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
