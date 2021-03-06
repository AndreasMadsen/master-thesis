
import tensorflow as tf

from code.tf_operator.decoder_residual_block \
    import parallel_decoder_residual_block


def parallel_bytenet_decoder(dec,
                             num_blocks=3, size=3,
                             rate=[1, 2, 4, 8, 16],
                             low_memory=False,
                             act='relu',
                             normalization='ln',
                             block_type='bytenet',
                             name=None, reuse=None):
    with tf.variable_scope(name, "bytenet-decoder", values=[dec], reuse=reuse):
        # loop dilated causal conv block
        for i in range(num_blocks):
            with tf.name_scope(f'bytenet-decoder-depth-{i}', values=[dec]):
                for rate_i in rate:
                    dec = parallel_decoder_residual_block(
                        dec, size=size, rate=rate_i,
                        low_memory=low_memory,
                        act=act,
                        normalization=normalization,
                        block_type=block_type,
                        name=f'decoder-res-block.{i}.{size}.{rate_i}'
                    )

        return dec
