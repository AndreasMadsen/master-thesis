
from nose.tools import assert_equal

import shutil
import os.path as path

import tensorflow as tf
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model.bytenet import ByteNet


def test_small_bytenet_on_digits():
    """ByteNet(version='v1-small-selu') works on single digits"""
    dataset = SyntheticDigits(
        examples=50, min_length=1, max_length=1,
        seed=99,
        tqdm=False
    )

    test_source = [
        'zero', 'one', 'two', 'three', 'four',
        'five', 'six', 'seven', 'eight', 'nine'
    ]
    test_expect = [
        '0^', '1^', '2^', '3^', '4^',
        '5^', '6^', '7^', '8^', '9^'
    ]

    if path.exists('asset/bytenet_local_quick_test'):
        shutil.rmtree('asset/bytenet_local_quick_test')

    stf.set_random_seed(99)
    model = ByteNet(dataset, num_blocks=1, latent_dim=20,
                    gpus=1, version='v1-small-selu',
                    save_dir='asset/bytenet_local_quick_test')
    model.train(max_ep=300, lr=0.05, tqdm=False)

    # hack to allow inference after Superviser Session is stoped
    tf.get_default_graph()._unsafe_unfinalize()

    test_greedy_predict = model.predict_from_str(test_source, reuse=True)
    assert_equal(list(test_greedy_predict), test_expect)

    test_beam_predict = model.predict_from_str(test_source,
                                               samples=5, reuse=True)
    assert_equal(list(test_beam_predict), test_expect)
