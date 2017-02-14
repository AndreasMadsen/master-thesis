
from typing import List

import numpy as np
import tensorflow as tf

from nltk.tokenize.moses import MosesTokenizer
from nltk.translate.bleu_score import corpus_bleu

from code.metric.abstract import Metric
from code.dataset.abstract import Dataset
from code.metric.util.nltk_env import NLTKEnv


def _tokenize_line(tokenizer, line):
    tokens = tokenizer.tokenize(line)
    last_token = tokens[-1]

    # split dot if the last token ends with .
    if last_token[-1] == '.' and len(last_token) > 1:
        tokens[-1] = last_token[:-1]
        tokens.append('.')

    return tokens


class BleuScore(Metric):
    _weights: List[float]

    def __init__(self, dataset: Dataset, name: str="BLEU-score", ngram=4):
        self._weights = [1/ngram]*ngram

        super().__init__(dataset, name=name)

    def _build_metric(self, model: 'code.model.abstract.Model') -> tf.Tensor:
        # predownload datasets
        with NLTKEnv() as nltk_env:
            nltk_env.download('perluniprops')
            nltk_env.download('nonbreaking_prefixes')

        with tf.name_scope(None, self.metric_name,
                           values=[self.dataset.source, self.dataset.target]):
            x = tf.cast(self.dataset.source, tf.int32)
            y = tf.cast(self.dataset.target, tf.int32)
            x = tf.Print(x, [x], 'x')
            predicted = model.inference_model(x, reuse=True)

            bleu = tf.py_func(self._py_implementaton,
                              [predicted, y],
                              tf.float32,
                              stateful=False,
                              name='nltk-corpus-bleu')
            return bleu

    def _py_implementaton(self, hypothesis, references):
        hypothesis = self.dataset.decode_as_batch(hypothesis, show_eos=False)
        references = self.dataset.decode_as_batch(references, show_eos=False)

        with NLTKEnv() as nltk_env:
            tokenizer = MosesTokenizer(lang=self.dataset.target_lang)

            # tokenize hypothesis references, and wrap each reference in an
            # array because there is only one reference.
            hypothesis_tokens = [
                _tokenize_line(tokenizer, line) for line in hypothesis
            ]
            references_tokens = [
                [_tokenize_line(tokenizer, line)] for line in references
            ]

            # calculate corpus-bleu score
            bleu = corpus_bleu(
                references_tokens, hypothesis_tokens,
                weights=self._weights
            )

            return np.asarray(bleu, dtype=np.float32)
