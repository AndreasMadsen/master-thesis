
import tensorflow as tf

from code.tf_operator.seq_dense import seq_dense
from code.tf_operator.bytenet_encoder import parallel_bytenet_encoder
from code.tf_operator.bytenet_decoder \
    import seq_bytenet_decoder, seq_bytenet_decoder_init
from code.tf_operator.embedding import embedding_matrix


def bytenet_unsupervised_translator(x,
                                    latent_dim=20, voca_size=20, num_blocks=3,
                                    rate=[1, 2, 4, 8, 16],
                                    labels=None, container=None,
                                    name=None, reuse=False):
    with tf.variable_scope(name, "bytenet-unsupervised-translator",
                           values=[x], reuse=reuse):
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
        enc = parallel_bytenet_encoder(enc,
                                       num_blocks=num_blocks, rate=rate,
                                       name="encoder")

        #
        # decode graph ( causal convolution )
        #
        # initalize scan state
        init_state = seq_bytenet_decoder_init(enc,
                                              num_blocks=num_blocks, rate=rate)

        # apply seq_decoder_residual_block to all time steps
        def scan_op(acc, enc_t):
            (state_tm1, logits_tm1, y_tm1) = acc

            # concat encoding at `t` and decoding at `t-1`
            dec = enc_t.sg_concat(target=y_tm1.sg_lookup(emb=emb_y))
            # decode graph ( causal convolution )
            state_t, dec = seq_bytenet_decoder(
                state_tm1, dec,
                num_blocks=num_blocks, rate=rate,
                name="decoder"
            )

            # final fully convolution layer for softmax
            logits_t = seq_dense(
                dec, dim=voca_size,
                name='logits-dense'
            )
            # get the most likely label
            label_t = tf.cast(tf.argmax(logits_t, axis=1), tf.int32)

            return (state_t, logits_t, label_t)

        (_, logits, labels) = tf.scan(
            scan_op,
            elems=tf.transpose(enc, perm=[1, 0, 2]),
            initializer=(
                init_state,
                tf.zeros(
                    (tf.shape(enc)[0], voca_size), dtype=enc.dtype
                ),  # logits
                tf.zeros(
                    (tf.shape(enc)[0], ), dtype=tf.int32
                )  # labels
            )
        )

        logits = tf.transpose(logits, perm=[1, 0, 2])
        labels = tf.transpose(labels, perm=[1, 0])

        return (logits, labels)
