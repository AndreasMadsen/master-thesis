
from nose.tools import assert_equal

from code.dataset.util.length_histogram import LengthHistogram


def test_histogram_properties():
    """check histogram update and properties"""
    # construct
    histogram = LengthHistogram()
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


def test_histogram_encoding():
    """check histogram encoding and decoding"""
    histogram = LengthHistogram()
    histogram.add(3)
    histogram.add(3)
    histogram.add(5)
    histogram.add(2)

    # encode
    assert_equal(histogram.encode(), [0, 0, 1, 2, 0, 1])

    # decode
    histogram_decoded = LengthHistogram(histogram.encode())
    assert_equal(histogram_decoded.encode(), histogram.encode())


def test_histogram_initalization():
    """check histogram initialization"""
    histogram = LengthHistogram()
    histogram.add(3)

    extra_histogram = LengthHistogram()
    assert_equal(extra_histogram.encode(), [0])


def test_histogram_extend():
    """check histogram extending"""
    histogram = LengthHistogram([0, 0, 1, 2, 0, 1])
    histogram_extend = histogram.extend(2)

    assert_equal(histogram.encode(), [0, 0, 1, 2, 0, 1])
    assert_equal(histogram_extend.encode(), [0, 0, 0, 0, 1, 2, 0, 1])


def test_histogram_partitioner():
    """check histogram partitioner"""
    # no issues
    histogram = LengthHistogram(
        [0, 0, 0, 1, 2, 0, 0, 0, 10, 20, 30, 0, 1, 0, 2, 0, 5, 10]
    )
    assert_equal(histogram.partition(min_size=10, min_width=4), [9, 13])

    # last partition [13:] is too small, and not included
    histogram = LengthHistogram(
        [0, 0, 0, 1, 2, 0, 0, 0, 10, 20, 30, 0, 1, 0, 2, 0, 5]
    )
    assert_equal(histogram.partition(min_size=10, min_width=4), [9])

    # min_width is longer than what is possibol
    histogram = LengthHistogram(
        [0, 0, 15]
    )
    assert_equal(histogram.partition(min_size=10, min_width=4), [])

    # min_size is less than observations
    histogram = LengthHistogram(
        [0, 0, 15, 5, 1, 5, 7]
    )
    assert_equal(histogram.partition(min_size=100, min_width=4), [])
