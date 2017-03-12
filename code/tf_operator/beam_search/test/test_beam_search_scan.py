
from nose.tools import assert_equal

import numpy as np
import tensorflow as tf

from code.tf_operator.beam_search.beam_search_scan import beam_search_scan
from code.tf_operator.select_value.batch_gather import batch_gather


def _makov_chain_scan_func(prev_tm1, elem_t):
    (state_tm1, logits_tm1, label_tm1) = prev_tm1

    state_t = (tf.range(tf.shape(state_tm1[0])[0]), )
    logits_t = tf.log(batch_gather(elem_t, label_tm1))

    return (state_t, logits_t)


def _test_output_switch():
    "validate output of beam_search_scan() on beam switch"
    transfer_matrix = np.asarray([[
        # beam picks [[2], [3]]
        [[0.01, 0.01, 0.49, 0.49],
         [0.97, 0.01, 0.01, 0.01],
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25]],

        # beam now sees that 3, is the better choice and goes to
        # [[3, 2], [3, 3]]
        [[0.97, 0.01, 0.01, 0.01],
         [0.97, 0.01, 0.01, 0.01],
         [0.25, 0.25, 0.25, 0.25],
         [0.01, 0.01, 0.49, 0.49]],

        # beam now moves to:
        # [[3, 2, 1], [3, 3, 3]]
        [[0.97, 0.01, 0.01, 0.01],
         [0.97, 0.01, 0.01, 0.01],
         [0.01, 0.97, 0.01, 0.01],
         [0.01, 0.01, 0.01, 0.97]],

        # beam manually moves from 1 to 0, overrulling the transfer matrix
        # beam does not move both paths to [3, 3, 3], even though they have
        # a higher properbility. This is because [3, 2, 1] higher (equal)
        # properbility to [3, 3, 3]
        # [[3, 2, 1, 0], [3, 3, 3, x]]
        [[0.97, 0.01, 0.01, 0.01],
         [0.01, 0.01, 0.49, 0.49],
         [0.97, 0.01, 0.01, 0.01],
         [0.25, 0.25, 0.25, 0.25]]
    ]], np.float32)

    with tf.Session() as sess:
        ret = beam_search_scan(
            _makov_chain_scan_func, beam_size=2,
            elems=transfer_matrix,
            initializer=(
                (tf.zeros(1, dtype=tf.int32), ),
                tf.zeros((1, 4), dtype=tf.float32),
                tf.zeros(1, dtype=tf.int32)
            )
        )

        (state_op, logprops_op, labels_op) = ret
        (state, logprops, labels) = sess.run(ret)

        assert_equal(state_op[0].get_shape(), (1, 2, 4))  # (ba, be, t)
        assert_equal(logprops_op.get_shape(), (1, 2, 4))  # (ba, be, t)
        assert_equal(labels_op.get_shape(), (1, 2, 4))  # (ba, be, t)

        np.testing.assert_almost_equal(
            state[0][0],
            np.asarray([
                [0, 0, 1, 1],
                [0, 0, 0, 0]
            ], dtype=np.int32)
        )

        np.testing.assert_almost_equal(
            labels[0],
            np.asarray([
                [3, 3, 3, 0],
                [3, 2, 1, 0]
            ], dtype=np.int32)
        )

        np.testing.assert_almost_equal(
            logprops[0],
            np.log(np.asarray([
                [0.49, 0.49**2, 0.49**2 * 0.97, 0.49**2 * 0.97 * 0.25],
                [0.49, 0.49**2, 0.49**2 * 0.97, 0.49**2 * 0.97 * 1.00]
            ], dtype=np.float32))
        )


def test_output_early_eneded():
    "validate output of beam_search_scan() on ended flag moves"
    transfer_matrix = np.asarray([[
        # beam picks [[2], [3], [1]], this marks beam 2 to ended
        [[0.01, 0.48, 0.03, 0.48],
         [0.97, 0.01, 0.01, 0.01],
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25]],

        # beam now sees that 3, is the better choice and goes to
        # [[3, 3], [1, 0], [3, 2]]
        # this moves beam 2 to beam 1, this the ended flag needs to be moved
        # if not [3, 2] will have ended, and the sequence becomes [3, 2, 0, 0]
        [[0.97, 0.01, 0.01, 0.01],
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25],
         [0.01, 0.01, 0.49, 0.49]],

        # beam now moves to:
        # [[3, 2, 1], [1, 0, 0], [3, 3, 1]]
        [[0.97, 0.01, 0.01, 0.01],
         [0.97, 0.01, 0.01, 0.01],
         [0.01, 0.97, 0.01, 0.01],
         [0.01, 0.97, 0.01, 0.01]],

        # [[3, 2, 1, 0], [1, 0, 0, 0], [3, 3, 1, 0]]
        [[0.97, 0.01, 0.01, 0.01],
         [0.97, 0.01, 0.01, 0.01],
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25]]
    ]], np.float32)

    with tf.Session() as sess:
        ret = beam_search_scan(
            _makov_chain_scan_func, beam_size=3,
            elems=transfer_matrix,
            initializer=(
                (tf.zeros(1, dtype=tf.int32), ),
                tf.zeros((1, 4), dtype=tf.float32),
                tf.zeros(1, dtype=tf.int32)
            )
        )

        (state_op, logprops_op, labels_op) = ret
        (state, logprops, labels) = sess.run(ret)

        assert_equal(state_op[0].get_shape(), (1, 3, 4))  # (ba, be, t)
        assert_equal(logprops_op.get_shape(), (1, 3, 4))  # (ba, be, t)
        assert_equal(labels_op.get_shape(), (1, 3, 4))  # (ba, be, t)

        np.testing.assert_almost_equal(
            labels[0],
            np.asarray([
                [3, 3, 1, 0],
                [1, 0, 0, 0],
                [3, 2, 1, 0]
            ], dtype=np.int32)
        )

        np.testing.assert_almost_equal(
            logprops[0],
            np.log(np.asarray([
                [0.03, 0.48 * 0.49, 0.48 * 0.49 * 0.97, 0.48 * 0.49 * 0.97],
                [0.48, 0.48 * 1.00, 0.48 * 1.00 * 1.00, 0.48 * 1.00 * 1.00],
                [0.48, 0.48 * 0.49, 0.48 * 0.49 * 0.97, 0.48 * 0.49 * 0.97]
            ], dtype=np.float32)),
            decimal=6
        )


def _test_output_idendity():
    "validate output of beam_search_scan() on idendity transfer matrix"
    transfer_matrix = np.asarray([[
        [[0.98, 0.01, 0.01],
         [0.01, 0.98, 0.01],
         [0.01, 0.01, 0.98]],
        [[0.98, 0.01, 0.01],
         [0.01, 0.98, 0.01],
         [0.01, 0.01, 0.98]],
        [[0.98, 0.01, 0.01],
         [0.01, 0.98, 0.01],
         [0.01, 0.01, 0.98]]
    ]], np.float32)

    with tf.Session() as sess:
        ret = beam_search_scan(
            _makov_chain_scan_func, beam_size=2,
            elems=transfer_matrix,
            initializer=(
                (tf.zeros(1, dtype=tf.int32), ),
                tf.zeros((1, 3), dtype=tf.float32),
                tf.zeros(1, dtype=tf.int32)
            )
        )

        (state_op, logprops_op, labels_op) = ret
        (state, logprops, labels) = sess.run(ret)

        assert_equal(state_op[0].get_shape(), (1, 2, 3))  # (ba, be, t)
        assert_equal(logprops_op.get_shape(), (1, 2, 3))  # (ba, be, t)
        assert_equal(labels_op.get_shape(), (1, 2, 3))  # (ba, be, t)

        np.testing.assert_almost_equal(
            state[0][0],
            np.asarray([
                [0, 0, 0],  # first beam (always), reused, reused
                [0, 1, 1]  # first beam (always), moved to beam 1, reuse
            ], dtype=np.float32)
        )

        np.testing.assert_almost_equal(
            labels[0],
            np.asarray([
                [1, 0, 0],  # 1, next most likely, <eos> causes <null>
                [0, 0, 0]  # 0 is most likely, no special rule
            ], dtype=np.int32)
        )

        np.testing.assert_almost_equal(
            logprops[0],
            np.log(np.asarray([
                [0.01, 0.01, 0.01],  # 1, next most likely, <eos> causes <null>
                [0.98, 0.98**2, 0.98**3]  # 0 is most likely, no special rule
            ], dtype=np.float32))
        )
