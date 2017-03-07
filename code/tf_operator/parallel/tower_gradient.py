
import tensorflow as tf

from code.tf_operator.math.mean_n import mean_n
from code.tf_operator.device.gpu_device import gpu_device


def tower_gradient(opt, losses, var_list):
    tower_grads = []
    for device_name, loss in losses:
        with gpu_device(device_name):
            tower_grads.append(opt.compute_gradients(loss, var_list=var_list))

    average_grads = []
    with tf.device('/cpu:0'):
        for grad_and_var in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   [(grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN)]
            grad_gpus, var_gpus = zip(*grad_and_var)

            # The Variables are shared across towers, so just return the
            # first tower's pointer to the Variable.
            average_grads.append((mean_n(grad_gpus), var_gpus[0]))

    return average_grads
