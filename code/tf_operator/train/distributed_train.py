
import os.path as path

import tensorflow as tf
import sugartensor as stf
from tensorflow.python.client import timeline as tf_timeline

from code.tf_operator.device.gpu_device import gpu_device


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


def distributed_train(**kwargs):
    opt = stf.sg_opt(kwargs)
    assert opt.loss is not None, 'loss is mandatory.'
    assert opt.distribution is not None, 'distribution is mandatory.'

    # default training options
    opt += stf.sg_opt(profile=0,
                      optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99)

    # create prefix <=> device mapping
    unmapped_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    unmapped_variables = set(tf.trainable_variables())

    category_prefixes = []
    for (category_name, device_name) in opt.distribution:
        prefix_set = set()

        match_scope = f'/{category_name}/'
        prefix_set.add(category_name)

        for t in tf.trainable_variables():
            if t.name.startswith(category_name):
                unmapped_variables.remove(t)

        for t in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
            if match_scope in t.name:
                update_op_prefix = t.name[:t.name.index(match_scope)]
                prefix_set.add(f'{update_op_prefix}/{category_name}')
                unmapped_update_ops.remove(t)

        category_prefixes.append(tuple(prefix_set))

    # validate all update_ops and variables mapped to a device
    assert len(unmapped_variables) == 0, 'not all variables are mapped'
    assert len(unmapped_update_ops) == 0, 'not all variables are mapped'

    # collect optimizer
    train_op = []

    for (category_name, device_name), category_prefixes \
            in zip(opt.distribution, category_prefixes):

        with gpu_device(device_name):
            train_op += stf.sg_optim(opt.loss, optim=opt.optim, lr=opt.lr,
                                     beta1=opt.beta1, beta2=opt.beta2,
                                     category=category_prefixes)

    profile_state = _ShouldProfile(opt.profile)

    # define train function
    @stf.sg_train_func
    def train_func(sess, arg):
        profile_state.increment()

        if profile_state.should_profile():
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            options = None
            run_metadata = None

        loss = sess.run([opt.loss] + train_op,
                        options=options,
                        run_metadata=run_metadata)[0]

        if profile_state.should_profile():
            tl = tf_timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(path.join(opt.save_dir, 'timeline.json'), 'w') as fd:
                print(ctf, file=fd)

        return loss

    # run train function
    train_func(**opt)
