
from typing import List

import sugartensor as stf
import numpy as np

from model.abstract.model import Model
from dataset.abstract.text_dataset import TextDataset
from tf_operator \
    import parallel_decoder_residual_block, parallel_encoder_residual_block


class ByteNet(Model):
    latent_dim: int
    num_blocks: int

    def __init__(self, dataset: TextDataset,
                 latent_dim: int=400, num_blocks: int=3) -> None:
        super().__init__(dataset)

        self.latent_dim = latent_dim
        self.num_blocks = num_blocks

    def _build_model(self, x: stf.Tensor, y_src: stf.Tensor) -> stf.Tensor:
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

        #
        # encode graph ( atrous convolution )
        #
        with stf.name_scope(None, "encoder", values=[x, emb_x]):

            # embed table lookup
            enc = x.sg_lookup(emb=emb_x)

            # loop dilated conv block
            for i in range(self.num_blocks):
                with stf.name_scope(None, "lyr-encoder", values=[enc]):
                    enc = parallel_encoder_residual_block(enc, size=5, rate=1)
                    enc = parallel_encoder_residual_block(enc, size=5, rate=2)
                    enc = parallel_encoder_residual_block(enc, size=5, rate=4)
                    enc = parallel_encoder_residual_block(enc, size=5, rate=8)
                    enc = parallel_encoder_residual_block(enc, size=5, rate=16)

        #
        # decode graph ( causal convolution )
        #
        with stf.name_scope(None, "decoder", values=[enc, y_src, emb_y]):
            dec = enc.sg_concat(target=y_src.sg_lookup(emb=emb_y))

            # loop dilated causal conv block
            for i in range(self.num_blocks):
                with stf.name_scope(None, "lyr-decoder", values=[dec]):
                    dec = parallel_decoder_residual_block(dec, size=3, rate=1)
                    dec = parallel_decoder_residual_block(dec, size=3, rate=2)
                    dec = parallel_decoder_residual_block(dec, size=3, rate=4)
                    dec = parallel_decoder_residual_block(dec, size=3, rate=8)
                    dec = parallel_decoder_residual_block(dec, size=3, rate=16)

            # final fully convolution layer for softmax
            return dec.sg_conv1d(size=1, dim=self.dataset.vocabulary_size)

    def train(self, max_ep=20):
        with stf.name_scope(None, "preprocessing",
                            values=[self.dataset.source, self.dataset.target]):
            # get source and target tensors
            x = stf.cast(self.dataset.source, stf.int32)
            y = stf.cast(self.dataset.target, stf.int32)

            # shift target for training source
            y_src = stf.concat(1, [
                # first value is zero
                stf.zeros((stf.shape(y)[0], 1), y.dtype),
                # skip last value
                y[:, :-1]
            ])

        with stf.name_scope(None, "model", values=[x, y_src]):
            dec = self._build_model(x, y_src)

        with stf.name_scope(None, "optimization", values=[dec, y]):
            # cross entropy loss with logit and mask
            loss = dec.sg_ce(target=y, mask=True)

        # train
        stf.sg_train(log_interval=30,
                     lr=0.0001,
                     loss=loss,
                     ep_size=self.dataset.num_batch,
                     max_ep=max_ep,
                     early_stop=False)

    def predict(self, sources) -> List[str]:
        sources = self.dataset.encode_as_batch(sources)
        predict_shape = (sources.shape[0], self.dataset.effective_max_length)

        # get source and target tensors
        x = stf.placeholder(dtype=stf.int32, shape=sources.shape)
        y_src = stf.placeholder(dtype=stf.int32, shape=predict_shape)

        dec = self._build_model(x, y_src)

        # greedy search policy
        label = dec.sg_argmax()

        # run graph for translating
        with stf.Session() as sess:
            # init session vars
            stf.sg_init(sess)

            # restore parameters
            saver = stf.train.Saver()
            saver.restore(
                sess,
                stf.train.latest_checkpoint('asset/train/ckpt')
            )

            # TODO: the original code has a strange for loop here: `range(3)`

            # initialize character sequence
            pred_prev = np.zeros(predict_shape, dtype=np.int32)
            pred = np.zeros(predict_shape, dtype=np.int32)

            # generate output sequence
            for i in range(self.dataset.effective_max_length):
                # predict character
                out = sess.run(label, {x: sources, y_src: pred_prev})
                # update character sequence
                if i < self.dataset.effective_max_length - 1:
                    pred_prev[:, i + 1] = out[:, i]
                pred[:, i] = out[:, i]

        return self.dataset.decode_as_batch(pred)
