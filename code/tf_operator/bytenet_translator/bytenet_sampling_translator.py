
import tensorflow as tf
import sugartensor as stf

from code.tf_operator.seq_dense import seq_dense
from code.tf_operator.bytenet_encoder import parallel_bytenet_encoder
from code.tf_operator.bytenet_decoder \
    import seq_bytenet_decoder, seq_bytenet_decoder_init
from code.tf_operator.batch_repeat \
    import batch_repeat, batch_repeat_pack, batch_repeat_unpack
from code.tf_operator.embedding import embedding_matrix


def bytenet_sampling_translator(x,
                                latent_dim=20, voca_size=20, num_blocks=3,
                                labels=None, container=None,
                                samples=1, seed=None,
                                name=None, reuse=False):
    with tf.variable_scope(name, "bytenet-sampling-translator",
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
        enc = parallel_bytenet_encoder(enc, num_blocks=num_blocks,
                                       name="encoder")

        # repeat encoding matrix for paralllel sampling
        # enc.shape = (batch * repeats, time, dims)
        enc = batch_repeat(enc, repeats=samples)
        enc = batch_repeat_pack(enc)

        #
        # decode graph ( causal convolution )
        #
        # initalize scan state
        init_state = seq_bytenet_decoder_init(enc, num_blocks=num_blocks)

        # apply seq_decoder_residual_block to all time steps
        def scan_op(acc, enc_t):
            (state_tm1, label_tm1, y_tm1) = acc

            # concat encoding at `t` and decoding at `t-1`
            dec = enc_t.sg_concat(target=y_tm1.sg_lookup(emb=emb_y))
            # decode graph ( causal convolution )
            state_t, dec = seq_bytenet_decoder(
                state_tm1, dec, num_blocks=num_blocks, name="decoder"
            )

            # final fully convolution layer for softmax
            logits_t = seq_dense(
                dec, dim=voca_size,
                name='logits-dense'
            )
            # sample from softmax distribution
            # this uses a binary search, the CPU time is:
            # O(batch * (num_samples * log(num_classes) + num_classes))
            label_t = tf.squeeze(tf.multinomial(logits_t, 1, seed=seed))
            label_t = tf.cast(label_t, dtype=tf.int32)

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
        logits = batch_repeat_unpack(logits, repeats=samples)
        labels = tf.transpose(labels, perm=[1, 0])
        labels = batch_repeat_unpack(labels, repeats=samples)

        return (logits, labels)
