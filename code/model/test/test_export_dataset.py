
from nose.tools import *

from code.dataset import SyntheticDigits
from code.model.export_dataset import ExportDataset


def test_export_dataset():
    """ExportDataset() collects the dataset"""
    dataset = SyntheticDigits(examples=10, shuffle=False, seed=99)
    export = ExportDataset(dataset)
    export.train()

    actual = list(zip(export.sources, export.targets))
    assert_equal(actual, [
        ('three eight eight', '388'),
        ('eight four nine', '849'),
        ('two eight', '28'),
        ('five four', '54'),
        ('seven three zero', '730'),
        ('seven zero four', '704'),
        ('one one nine', '119'),
        ('four one', '41'),
        ('six seven one', '671'),
        ('four zero seven', '407')
    ])


def test_export_dataset_show_eos():
    """ExportDataset(show_eos=True) collects the dataset"""
    dataset = SyntheticDigits(examples=10, shuffle=False, seed=99)
    export = ExportDataset(dataset, show_eos=True)
    export.train()

    actual = list(zip(export.sources, export.targets))
    assert_equal(actual, [
        ('three eight eight#', '388#'),
        ('eight four nine#', '849#'),
        ('two eight#', '28#'),
        ('five four#', '54#'),
        ('seven three zero#', '730#'),
        ('seven zero four#', '704#'),
        ('one one nine#', '119#'),
        ('four one#', '41#'),
        ('six seven one#', '671#'),
        ('four zero seven#', '407#')
    ])
