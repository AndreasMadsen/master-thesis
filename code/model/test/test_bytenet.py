
from nose.tools import *

import shutil
import os.path as path

import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model.bytenet import ByteNet
from code.model.export_dataset import ExportDataset


def test_bytenet_on_digits():
    """ByteNet() works on single digits"""
    dataset = SyntheticDigits(
        examples=50, min_length=1, max_length=1,
        seed=99
    )

    test_source = [
        'zero', 'one', 'two', 'three', 'four',
        'five', 'six', 'seven', 'eight', 'nine'
    ]
    test_expect = [
        '0#', '1#', '2#', '3#', '4#',
        '5#', '6#', '7#', '8#', '9#'
    ]

    if path.exists('asset/bytenet-local-quick-test'):
        shutil.rmtree('asset/bytenet-local-quick-test')

    stf.set_random_seed(99)
    model = ByteNet(dataset, num_blocks=1, latent_dim=20,
                    save_dir='asset/bytenet-local-quick-test')
    model.train(max_ep=200, lr=0.1, tqdm=False)
    test_predict = model.predict(test_source, reuse=True)

    assert_equal(test_predict, test_expect)