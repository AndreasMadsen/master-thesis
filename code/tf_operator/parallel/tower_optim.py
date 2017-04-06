
import sugartensor as stf
import tensorflow as tf

from code.tf_operator.parallel.tower_gradient import tower_gradient
from code.tf_operator.train.optimizer import optimizer


def _find_variable(op):
    def recursive(nested_op):
        if nested_op.type == 'VariableV2':
            yield nested_op
        else:
            for inp in nested_op.inputs:
                yield from recursive(inp.op)

    # By pure change the variable that we want is the first,
    # this could be quite error prone.
    variable = next(recursive(op))

    return variable


def tower_optim(losses, summary=None, **kwargs):
    # get options
    opt = stf.sg_opt(summary=summary) + stf.sg_get_context() \
                                      + stf.sg_opt(summary=True)

    # get optimizer
    optim = optimizer(**kwargs)

    # get trainable variables
    var_list = tf.trainable_variables()

    # calc gradients (represented as (grad, var) pairs) like compute_gradients
    gradient = tower_gradient(optim, losses, var_list)

    # add summary
    if opt.summary:
        for g, v in gradient:
            # exclude batch normal statics
            if 'mean' not in v.name and 'variance' not in v.name \
                    and 'beta' not in v.name and 'gamma' not in v.name:
                stf.sg_summary_gradient(v, gradient=g)

    # gradient update op
    if len(losses) > 1:
        with tf.device('/cpu:0'):
            grad_op = optim.apply_gradients(
                gradient, global_step=stf.sg_global_step()
            )
    else:
        grad_op = optim.apply_gradients(
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
        variable = _find_variable(tensor.op)
        variable_name = variable.name.split('/')[-1]

        # because the variable search is quite hackish, validate that
        # the update op name is correct.
        if variable_name not in [
            'moving_variance', 'moving_mean', 'variance', 'mean'
        ]:
            raise TypeError(
                f'invalid variable {variable.name}' +
                f' ({variable_name}) found in {tensor.name}'
            )

        # get the name of the variable that the assign op will change
        unique_update_op[variable.name] = tensor
    update_op = list(unique_update_op.values())

    return [grad_op] + update_op
