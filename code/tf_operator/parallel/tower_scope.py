
import tensorflow as tf

from code.tf_operator.device.gpu_device import gpu_device


def _cpu_variable_allocator(getter, *args, **kwargs):
    with tf.device('/cpu:0'):
        return getter(*args, **kwargs)


def tower_scope(gpus=[0], reuse=None):
    for gpu_index in gpus:
        device_name = f'/gpu:{gpu_index}'
        with tf.variable_scope(tf.get_variable_scope(),
                               custom_getter=_cpu_variable_allocator,
                               reuse=reuse), gpu_device(device_name):
            reuse = True
            yield gpu_index, device_name
