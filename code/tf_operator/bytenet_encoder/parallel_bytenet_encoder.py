
import tensorflow as tf

from code.tf_operator.encoder_residual_block \
    import parallel_encoder_residual_block


def parallel_bytenet_encoder(x,
                             num_blocks=3, size=5,
                             rate=[1, 2, 4, 8, 16],
                             normalization='bn',
                             name=None, reuse=None):
    with tf.variable_scope(name, "bytenet-encoder", values=[x], reuse=reuse):
        enc = x

        # loop dilated conv block
        for i in range(num_blocks):
            with tf.name_scope(f'bytenet-encoder-depth-{i}', values=[enc]):
                for rate_i in rate:
                    enc = parallel_encoder_residual_block(
                        enc, size=size, rate=rate_i,
                        normalization=normalization,
                        name=f'encoder-res-block.{i}.{size}.{rate_i}'
                    )

        return enc
