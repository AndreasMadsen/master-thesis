
from nose.tools import assert_almost_equals

import os.path as path

import tensorflow as tf
import sugartensor as stf

from code.metric.bleu_score import BleuScore
from code.dataset import WMTBilingualNews
from code.metric.test.dummy_model import DummyModel
from code.metric.test.dummy_dataset import DummyDataset


def _load_fixture(name):
    # read google translated lines
    this_dir = path.dirname(path.realpath(__file__))
    filepath = path.join(this_dir, f'fixtures/{name}.txt')
    with open(filepath, encoding='utf-8') as file:
        translated_lines = [line.rstrip() for line in file]
    return translated_lines


def test_blue_score_on_google():
    """test bleu score metric on google translated output"""
    dataset = WMTBilingualNews(
        year=2015, source_lang='en', target_lang='ru',
        batch_size=100, max_observations=100,
        min_length=0, max_length=1024,
        shuffle=False
    )

    # encode translated text
    translated = dataset.encode_as_batch(_load_fixture('trans.google.ru'))

    # setup model
    model = DummyModel(dataset, translated)
    bleu_2gram = BleuScore(dataset, ngram=2).build(model)
    bleu_4gram = BleuScore(dataset).build(model)

    with tf.Session() as sess:
        stf.sg_init(sess)
        with stf.sg_queue_context():
            score_2gram = sess.run(bleu_2gram)
            score_4gram = sess.run(bleu_4gram)

            # TODO: these are not quite correct because of the tokenizer,
            # see https://github.com/nltk/nltk/issues/1330 for what the
            # actual values should be.
            # NOTE: dataset.vocabulary does not contain all chars from
            # translated, which also causes some discrepancies.
            assert_almost_equals(score_2gram, 40.008049, places=6)
            assert_almost_equals(score_4gram, 23.189312, places=6)


test_blue_score_on_google()


def test_blue_score_on_poorly():
    """test bleu score metric on poorly translated output"""
    dataset = DummyDataset(
        _load_fixture('ref.poor.en'),
        source_lang='fr', target_lang='en',
        batch_size=128,
        shuffle=False
    )

    # encode translated text
    translated = dataset.encode_as_batch(_load_fixture('trans.poor.en'))

    # setup model
    model = DummyModel(dataset, translated)
    bleu_4gram = BleuScore(dataset).build(model)

    with tf.Session() as sess:
        stf.sg_init(sess)
        with stf.sg_queue_context():
            score_4gram = sess.run(bleu_4gram)
            assert_almost_equals(score_4gram, 0.0, places=1)
