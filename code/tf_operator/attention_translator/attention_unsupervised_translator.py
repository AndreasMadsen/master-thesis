
import tensorflow as tf
import sugartensor as stf
from tensorflow.contrib import rnn, seq2seq

from code.tf_operator.attention_translator.attention_encoder \
    import attention_encoder
from code.tf_operator.attention_translator.attention_decoder \
    import attention_decoder
from code.tf_operator.embedding import embedding_matrix


def attention_unsupervised_translator(x, length,
                                      latent_dim=20, voca_size=20,
                                      num_blocks=3,
                                      max_length=None,
                                      labels=None, container=None,
                                      name=None, reuse=None):
    with tf.variable_scope(name, "attention-unsupervised-translator",
                           values=[x, length], reuse=reuse):
        # make embedding matrix for source and target
        emb_x = embedding_matrix(
            voca_size=voca_size,
            dim=latent_dim,
            name='embedding-source',
            labels=labels, container=container
        )
        emb_y = embedding_matrix(
            voca_size=voca_size,
            dim=latent_dim,
            name='embedding-target',
            labels=labels, container=container
        )

        # encode graph ( atrous convolution )
        enc = x.sg_lookup(emb=emb_x)
        enc = attention_encoder(enc, length,
                                num_blocks=num_blocks, name="encoder")

        def unsupervised_train_helper():
            # get shapes
            batch_size = x.get_shape().as_list()[0]
            if batch_size is None:
                batch_size = tf.shape(x)[0]

            return seq2seq.GreedyEmbeddingHelper(
                lambda ids: ids.sg_lookup(emb=emb_y),
                start_tokens=tf.fill([batch_size], 0),
                end_token=1)

        logits, labels = attention_decoder(enc, length,
                                           unsupervised_train_helper,
                                           voca_size=voca_size,
                                           max_length=max_length,
                                           name="decoder")

        return (logits, labels)
