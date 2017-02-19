
import numpy as np
import tensorflow as tf

from code.metric.abstract import Metric
from code.dataset.abstract import Dataset
from code.metric.util.nltk_env import NLTKEnv
from code.metric.calculate.multi_bleu import multi_bleu


def _tokenize_line(tokenizer, line):
    tokens = tokenizer.tokenize(line)
    last_token = tokens[-1]

    # split dot if the last token ends with .
    if last_token[-1] == '.' and len(last_token) > 1:
        tokens[-1] = last_token[:-1]
        tokens.append('.')

    return tokens


class BleuScore(Metric):
    _ngram: int

    def __init__(self, dataset: Dataset, name: str="BLEU-score", ngram=4):
        self._ngram = ngram

        super().__init__(dataset, name=name)

    def _build_metric(self, model: 'code.model.abstract.Model') -> tf.Tensor:
        # predownload datasets
        with NLTKEnv() as nltk_env:
            nltk_env.download('perluniprops')
            nltk_env.download('nonbreaking_prefixes')

        with tf.name_scope(None, self.metric_name,
                           values=[self.dataset.source, self.dataset.target]):
            predicted = model.inference_model(self.dataset.source, reuse=True)

            bleu = tf.py_func(self._py_implementaton,
                              [predicted, self.dataset.target],
                              tf.float32,
                              stateful=False,
                              name='nltk-corpus-bleu')
            return bleu

    def _py_implementaton(self, hypothesis, references):
        hypothesis = self.dataset.decode_as_batch(hypothesis, show_eos=False)
        references = self.dataset.decode_as_batch(references, show_eos=False)

        with NLTKEnv() as nltk_env:
            from nltk.tokenize.moses import MosesTokenizer

            tokenizer = MosesTokenizer(lang=self.dataset.target_lang)

            # tokenize hypothesis references, and wrap each reference in an
            # array because there is only one reference.
            hypothesis_tokens = [
                _tokenize_line(tokenizer, line) for line in hypothesis
            ]
            references_tokens = [
                _tokenize_line(tokenizer, line) for line in references
            ]

            # calculate corpus-bleu score
            score, precisions, brevity_penalty, _, _ = multi_bleu(
                hypothesis_tokens, [references_tokens],
                maxn=self._ngram
            )

            return np.asarray(score, dtype=np.float32)
