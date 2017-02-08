
import tensorflow as tf
import sugartensor as stf

from code.tf_operator.bytenet_encoder import parallel_bytenet_encoder
from code.tf_operator.bytenet_decoder import parallel_bytenet_decoder


def bytenet_supervised_translator(x, y,
                                  latent_dim=20, voca_size=20, num_blocks=3,
                                  name=None, reuse=False):
    with tf.variable_scope(name, "bytenet-supervised-translator",
                           values=[x, y], reuse=reuse):
        # make embedding matrix for source and target
        emb_x = stf.sg_emb(
            name='embedding-source',
            voca_size=voca_size,
            dim=latent_dim
        )
        emb_y = stf.sg_emb(
            name='embedding-target',
            voca_size=voca_size,
            dim=latent_dim
        )

        # shift target for training source
        with tf.name_scope("shift-target", values=[y]):
            y_src = tf.concat(1, [
                # first value is zero
                tf.zeros((stf.shape(y)[0], 1), y.dtype),
                # skip last value
                y[:, :-1]
            ])

        # encode graph ( atrous convolution )
        enc = x.sg_lookup(emb=emb_x)
        enc = parallel_bytenet_encoder(enc, num_blocks=num_blocks,
                                       name="encoder")

        # decode graph ( causal convolution )
        dec = enc.sg_concat(target=y_src.sg_lookup(emb=emb_y))
        dec = parallel_bytenet_decoder(dec, num_blocks=num_blocks,
                                       name="decoder")

        # final fully convolution layer for softmax
        logits = dec.sg_conv1d(
            size=1, dim=voca_size,
            name='logits-dense'
        )
        # get the most likely label
        labels = tf.cast(tf.argmax(logits, axis=2), tf.int32)

        return (logits, labels)
