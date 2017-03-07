
from contextlib import contextmanager
import shutil

import tensorflow as tf

gpu_available = shutil.which("nvidia-smi") is not None


@contextmanager
def gpu_device(device_name):
    if not gpu_available:
        device_name = '/cpu:0'

    with tf.device(device_name):
        yield
