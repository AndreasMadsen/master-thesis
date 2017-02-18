
from nose.tools import assert_almost_equals

import os.path as path

import tensorflow as tf
import sugartensor as stf

from code.model.abstract import Model
from code.metric.bleu_score import BleuScore
from code.dataset import WMTBilingualNews
from code.metric.test.dummy_model import DummyModel


def test_out_of_bound():
    """test bleu score metric on google translated output"""
    dataset = WMTBilingualNews(
        year=2015, source_lang='en', target_lang='ru',
        batch_size=100, observations=100,
        min_length=0, max_length=1024,
        shuffle=False
    )

    # read google translated lines
    this_dir = path.dirname(path.realpath(__file__))
    filepath = path.join(this_dir, 'fixtures/raw.google.ru.txt')
    with open(filepath, encoding='utf-8') as file:
        translated_lines = [line.rstrip() for line in file]

    # encode translated text
    translated = dataset.encode_as_batch(translated_lines)

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
            assert_almost_equals(score_2gram, 39.987335, places=6)
            assert_almost_equals(score_4gram, 23.176613, places=6)
