
from nose.tools import assert_equal

from code.dataset.util.dataset_properties_cache import CorpusProperties
from code.dataset.wmt_bilingual_news import WMTBilingualNews


def test_properties_2013():
    """check properties on WMTBilingualNews(fr, en, 2013)"""
    dataset_2013 = WMTBilingualNews(
        source_lang='fr', target_lang='en', year=2013,
        tqdm=False
    )
    assert_equal(dataset_2013.properties.histogram.observations, 1531)
    assert_equal(len(dataset_2013.properties.vocabulary), 117)


def test_properties_2014():
    """check properties on WMTBilingualNews(fr, en, 2014)"""
    dataset_2014 = WMTBilingualNews(
        source_lang='fr', target_lang='en', year=2014,
        tqdm=False
    )

    assert_equal(dataset_2014.properties.histogram.observations, 1373)
    assert_equal(len(dataset_2014.properties.vocabulary), 106)


def test_properties_2013_reload():
    """check reloadability on WMTBilingualNews(fr, en, 2013)"""
    dataset_2013 = WMTBilingualNews(
        source_lang='fr', target_lang='en', year=2013,
        tqdm=False
    )
    dataset_2013_reload = WMTBilingualNews(
        source_lang='fr', target_lang='en', year=2013,
        tqdm=False
    )

    assert_equal(dataset_2013.properties, dataset_2013_reload.properties)
