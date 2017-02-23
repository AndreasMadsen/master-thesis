
import tensorflow as tf
import sugartensor as stf

from code.tf_operator.bytenet_encoder import parallel_bytenet_encoder
from code.tf_operator.bytenet_decoder import parallel_bytenet_decoder
from code.tf_operator.embedding import embedding_matrix


def bytenet_supervised_translator(x, y,
                                  shift=True,
                                  latent_dim=20, voca_size=20, num_blocks=3,
                                  rate=[1, 2, 4, 8, 16],
                                  low_memory=False,
                                  labels=None, container=None,
                                  name=None, reuse=False):
    with tf.variable_scope(name, "bytenet-supervised-translator",
                           values=[x, y], reuse=reuse):
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

        # shift target for training source
        if shift:
            with tf.name_scope("shift-target", values=[y]):
                y_src = tf.concat([
                    # first value is zero
                    tf.zeros((stf.shape(y)[0], 1), y.dtype),
                    # skip last value
                    y[:, :-1]
                ], 1)
        else:
            y_src = y

        # encode graph ( atrous convolution )
        enc = x.sg_lookup(emb=emb_x)
        enc = parallel_bytenet_encoder(enc, num_blocks=num_blocks, rate=rate,
                                       name="encoder")

        # decode graph ( causal convolution )
        dec = enc.sg_concat(target=y_src.sg_lookup(emb=emb_y))
        dec = parallel_bytenet_decoder(dec, num_blocks=num_blocks, rate=rate,
                                       low_memory=low_memory,
                                       name="decoder")

        # final fully convolution layer for softmax
        logits = dec.sg_conv1d(
            size=1, dim=voca_size,
            name='logits-dense'
        )
        # get the most likely label
        labels = tf.cast(tf.argmax(logits, axis=2), tf.int32)

        return (logits, labels)
