
from nose.tools import assert_equal

import numpy as np
import tensorflow as tf
import sugartensor as stf

from code.tf_operator.bytenet_translator.bytenet_unsupervised_translator \
    import bytenet_unsupervised_translator
from code.tf_operator.bytenet_translator.bytenet_supervised_translator \
    import bytenet_supervised_translator


def naive_translator(sess, supervised, sources, x, y_tmp, vocab_size):
    # initialize character sequence
    pred_prev = np.zeros(sources.shape).astype(np.int32)
    pred = np.zeros(sources.shape).astype(np.int32)
    logits = np.zeros((sources.shape[0], sources.shape[1], vocab_size))

    # generate output sequence
    for i in range(sources.shape[1]):
        # predict character
        step_logits, step_labels = sess.run(supervised,
                                            {x: sources, y_tmp: pred_prev})

        # update character sequence
        if i < sources.shape[1] - 1:
            pred_prev[:, i + 1] = step_labels[:, i]
        logits[:, i] = step_logits[:, i]
        pred[:, i] = step_labels[:, i]

    return logits, pred


def test_bytenet_unsupervised_translator():
    "validate output bytenet_unsupervised_translator() matches naive inference"

    sources = np.asarray([
        [2, 4, 3, 6, 3, 3, 5, 1, 0, 0],
        [3, 5, 6, 8, 3, 6, 8, 9, 2, 1]
    ], dtype=np.int32)

    x = stf.placeholder(dtype=tf.int32, shape=sources.shape)
    y_tmp = stf.placeholder(dtype=tf.int32, shape=sources.shape)

    tf.set_random_seed(10)
    supervised = bytenet_supervised_translator(
        x, y_tmp, shift=False,
        voca_size=10,
        latent_dim=6,
        num_blocks=1,
        rate=[1, 2],
        name="bytenet-model"
    )
    unsupervised = bytenet_unsupervised_translator(
        x,
        voca_size=10,
        latent_dim=6,
        num_blocks=1,
        rate=[1, 2],
        name="bytenet-model", reuse=True
    )

    with tf.Session() as sess:
        stf.sg_init(sess)

        naive_logits, naive_labels = naive_translator(
            sess, supervised, sources, x, y_tmp, vocab_size=10
        )

        logits, label = sess.run(unsupervised, {x: sources})

        np.testing.assert_almost_equal(naive_logits, logits, decimal=6)
        np.testing.assert_almost_equal(naive_labels, label, decimal=6)
