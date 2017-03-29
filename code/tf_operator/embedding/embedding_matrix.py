
from typing import List

import tensorflow as tf
import sugartensor as stf

from code.tf_operator.embedding.embedding_container import EmbeddingContainer


def embedding_matrix(voca_size: int, dim: int,
                     name: str=None,
                     container: EmbeddingContainer=None,
                     summary=None,
                     labels: List[str]=None):
    opt = stf.sg_opt(summary=summary) + stf.sg_get_context() \
                                      + stf.sg_opt(summary=True)

    with tf.name_scope(None, 'embedding-matrix'):
        # initialize embedding matrix
        w = stf.sg_initializer.he_uniform(name, (voca_size - 1, dim),
                                          summary=opt.summary)
        if (container is not None and labels is not None):
            # since w is post-padded with a zero row, remove the first label
            container.add(w, labels[1:])

        # 1st row should be zero and not be updated by backprop because of
        # zero padding.
        emb = tf.concat([tf.zeros((1, dim), dtype=tf.float32), w], 0)

        return emb
