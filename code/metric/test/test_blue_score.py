
from nose.tools import assert_almost_equals

import os.path as path

import tensorflow as tf
import sugartensor as stf

from code.model.abstract import Model
from code.metric.bleu_score import BleuScore
from code.dataset import WMTBilingualNews


class DummyModel(Model):
    def __init__(self, dataset,
                 save_dir='asset/blue-score-test',
                 **kwargs):
        super().__init__(dataset, save_dir=save_dir, **kwargs)

        # read google translated lines
        this_dir = path.dirname(path.realpath(__file__))
        with open(path.join(this_dir, 'fixtures/raw.google.ru.txt')) as file:
            translated_lines = [line.rstrip() for line in file]

        # encode translated text
        self._tranlated = self.dataset.encode_as_batch(translated_lines)

    def inference_model(self, x, reuse=False):
        observations = int(self.dataset.target.get_shape()[0])

        translated = tf.convert_to_tensor(self._tranlated)
        translated = tf.train.slice_input_producer(
            [translated], shuffle=False
        )[0]
        translated = tf.train.batch(
            [translated], observations,
            name='inference',
            num_threads=1,
            capacity=observations,
            allow_smaller_final_batch=False
        )

        return translated


def test_bleu_score():
    """test bleu score metric on google translated output"""
    dataset = WMTBilingualNews(
        year=2015, source_lang='en', target_lang='ru',
        observations=100, min_length=0, max_length=1024,
        shuffle=False
    )

    model = DummyModel(dataset)
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
            assert_almost_equals(score_2gram, 0.39987335)
            assert_almost_equals(score_4gram, 0.23176613)
