
from contextlib import contextmanager
import os

import tensorflow as tf
jit_scope_context = tf.contrib.compiler.jit.experimental_jit_scope


@contextmanager
def jit_scope():
    if 'TF_USE_XLA' in os.environ:
        with jit_scope_context():
            yield
    else:
        yield
