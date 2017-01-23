
from nose.tools import *

from dataset import SyntheticDigits
from model.export_dataset import ExportDataset


def test_export_dataset():
    """ExportDataset() collects the dataset"""
    dataset = SyntheticDigits(examples=10, consistent=True, seed=99)
    export = ExportDataset(dataset)
    export.train()

    actual = list(zip(export.sources, export.targets))
    assert_equal(actual, [
        ('five four', '54'),
        ('eight four nine', '849'),
        ('four zero seven', '407'),
        ('four one', '41'),
        ('seven three zero', '730'),
        ('six seven one', '671'),
        ('three eight eight', '388'),
        ('seven zero four', '704'),
        ('two eight', '28'),
        ('one one nine', '119')
    ])


def test_export_dataset_show_eos():
    """ExportDataset(show_eos=True) collects the dataset"""
    dataset = SyntheticDigits(examples=10, consistent=True, seed=99)
    export = ExportDataset(dataset, show_eos=True)
    export.train()

    actual = list(zip(export.sources, export.targets))
    assert_equal(actual, [
        ('five four#', '54#'),
        ('eight four nine#', '849#'),
        ('four zero seven#', '407#'),
        ('four one#', '41#'),
        ('seven three zero#', '730#'),
        ('six seven one#', '671#'),
        ('three eight eight#', '388#'),
        ('seven zero four#', '704#'),
        ('two eight#', '28#'),
        ('one one nine#', '119#')
    ])
