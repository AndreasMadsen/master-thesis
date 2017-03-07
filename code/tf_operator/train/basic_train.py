
import os.path as path

import tensorflow as tf
import sugartensor as stf
from tensorflow.python.client import timeline as tf_timeline


class _ShouldProfile:
    iterations: int = 0
    want_iterations: int = 0
    enable: bool = False

    def __init__(self, profile: int):
        if profile > 0:
            self.enable = True
            self.want_iterations = profile

    def increment(self):
        self.iterations += 1

    def should_profile(self):
        return self.iterations == self.want_iterations and self.enable


def basic_train(loss_op, update_op,
                profile=0, save_dir='asset/unamed',
                **kwargs):
    profile_state = _ShouldProfile(profile)

    @stf.sg_train_func
    def train_func(sess, arg):
        profile_state.increment()

        if profile_state.should_profile():
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            options = None
            run_metadata = None

        loss = sess.run([loss_op] + update_op,
                        options=options,
                        run_metadata=run_metadata)[0]

        if profile_state.should_profile():
            tl = tf_timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(path.join(save_dir, 'timeline.json'), 'w') as fd:
                print(ctf, file=fd)

        return loss

    # run train function
    train_func(save_dir=save_dir, **kwargs)
