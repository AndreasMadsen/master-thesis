
from nose.tools import assert_equal

from code.dataset.util.length_histogram import LengthHistogram


def test_length_histogram():
    """check histogram writing, properties, and reloadability"""
    # construct
    histogram = LengthHistogram()

    # add
    histogram.add(3)
    histogram.add(3)
    histogram.add(5)
    histogram.add(2)

    # index
    assert_equal(histogram[1], 0)
    assert_equal(histogram[2], 1)
    assert_equal(histogram[3], 2)
    assert_equal(histogram[4], 0)
    assert_equal(histogram[5], 1)

    # length
    assert_equal(histogram.min_length, 2)
    assert_equal(histogram.max_length, 5)

    # observations
    assert_equal(histogram.observations, 4)

    # slice
    assert_equal(histogram[2:4], 3)
    assert_equal(histogram[2:5], 3)
    assert_equal(histogram[2:], 4)

    # encode
    assert_equal(histogram.encode(), [0, 0, 1, 2, 0, 1])

    # decode
    histogram_decoded = LengthHistogram(histogram.encode())
    assert_equal(histogram_decoded.encode(), histogram.encode())

    # no initial state
    extra_histogram = LengthHistogram()
    assert_equal(extra_histogram.encode(), [0])
