
from typing import List

import tensorflow as tf
import sugartensor as stf
import numpy as np

from code.model.abstract.model import Model
from code.dataset.abstract.text_dataset import TextDataset
from code.tf_operator import \
    seq_dense, \
    parallel_bytenet_decoder, parallel_bytenet_encoder, \
    seq_bytenet_decoder_init, seq_bytenet_decoder


class ByteNet(Model):
    latent_dim: int
    num_blocks: int
    _save_dir: str

    def __init__(self, dataset: TextDataset,
                 latent_dim: int=400, num_blocks: int=3,
                 save_dir='asset/bytenet') -> None:
        super().__init__(dataset)

        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self._save_dir = save_dir

    def _build_train_model(self,
                           x: tf.Tensor, y: tf.Tensor,
                           reuse=False) -> tf.Tensor:
        with tf.variable_scope("bytenet-model", values=[x, y], reuse=reuse):
            # make embedding matrix for source and target
            emb_x = stf.sg_emb(
                name='embedding-source',
                voca_size=self.dataset.vocabulary_size,
                dim=self.latent_dim
            )
            emb_y = stf.sg_emb(
                name='embedding-target',
                voca_size=self.dataset.vocabulary_size,
                dim=self.latent_dim
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
            enc = parallel_bytenet_encoder(enc, name="encoder")

            # decode graph ( causal convolution )
            dec = enc.sg_concat(target=y_src.sg_lookup(emb=emb_y))
            dec = parallel_bytenet_decoder(dec, name="decoder")

            # final fully convolution layer for softmax
            return dec.sg_conv1d(
                size=1, dim=self.dataset.vocabulary_size,
                name='logits-dense'
            )

    def _build_test_model(self,
                          x: tf.Tensor,
                          reuse=False) -> tf.Tensor:
        with tf.variable_scope("bytenet-model", values=[x], reuse=reuse):
            # make embedding matrix for source and target
            emb_x = stf.sg_emb(
                name='embedding-source',
                voca_size=self.dataset.vocabulary_size,
                dim=self.latent_dim
            )
            emb_y = stf.sg_emb(
                name='embedding-target',
                voca_size=self.dataset.vocabulary_size,
                dim=self.latent_dim
            )

            # encode graph ( atrous convolution )
            enc = x.sg_lookup(emb=emb_x)
            enc = parallel_bytenet_encoder(enc, name="encoder")

            #
            # decode graph ( causal convolution )
            #
            # initalize scan state
            init_state = seq_bytenet_decoder_init(enc)

            # apply seq_decoder_residual_block to all time steps
            def scan_op(acc, enc_t):
                (state_tm1, y_tm1) = acc

                # concat encoding at `t` and decoding at `t-1`
                dec = enc_t.sg_concat(target=y_tm1.sg_lookup(emb=emb_y))
                # decode graph ( causal convolution )
                state_t, dec = seq_bytenet_decoder(
                    state_tm1, dec, name="decoder"
                )

                # final fully convolution layer for softmax
                logits_t = seq_dense(
                    dec, dim=self.dataset.vocabulary_size,
                    name='logits-dense'
                )
                # get the most likely label
                label_t = tf.cast(tf.argmax(logits_t, axis=1), tf.int32)

                return (state_t, label_t)

            (_, labels) = tf.scan(
                scan_op,
                elems=tf.transpose(enc, perm=[1, 0, 2]),
                initializer=(
                    init_state,
                    tf.zeros(
                        (tf.shape(enc)[0], ), dtype=tf.int32
                    )  # labels
                )
            )

            return tf.transpose(labels, perm=[1, 0])

    def train(self, max_ep=20, lr=0.0001, **kwargs):
        with tf.name_scope(None, "preprocessing",
                           values=[self.dataset.source, self.dataset.target]):
            # get source and target tensors
            x = tf.cast(self.dataset.source, tf.int32)
            y = tf.cast(self.dataset.target, tf.int32)

        dec = self._build_train_model(x, y)

        with tf.name_scope(None, "optimization", values=[dec, y]):
            # cross entropy loss with logit and mask
            loss = dec.sg_ce(target=y, mask=True)

        # train
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=sess_config) as sess:
            stf.sg_train(log_interval=30,
                         lr=lr,
                         loss=loss,
                         ep_size=self.dataset.num_batch,
                         max_ep=max_ep,
                         early_stop=False,
                         save_dir=self._save_dir,
                         sess=sess,
                         **kwargs)

    def predict(self, sources, reuse=False) -> List[str]:
        sources = self.dataset.encode_as_batch(sources)
        predict_shape = (sources.shape[0], self.dataset.effective_max_length)

        # get source and target tensors
        x = stf.placeholder(dtype=stf.int32, shape=sources.shape)

        label = self._build_test_model(x, reuse=reuse)

        # run graph for translating
        with tf.Session() as sess:
            # init session vars
            stf.sg_init(sess)

            # restore parameters
            stf.sg_restore(sess, tf.train.latest_checkpoint(self._save_dir))

            pred = sess.run(label, {x: sources})

        return self.dataset.decode_as_batch(pred)
