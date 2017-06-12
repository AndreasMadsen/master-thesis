
import tensorflow as tf

from code.tf_operator.decoder_residual_block \
    import seq_decoder_residual_block_init, seq_decoder_residual_block


def seq_bytenet_decoder_init(enc, name=None,
                             num_blocks=3, size=3,
                             rate=[1, 2, 4, 8, 16],
                             block_type='bytenet'):
    with tf.name_scope(name, "decoder-scan-init", values=[enc]):
        latent_dim = int(enc.get_shape()[-1])

        init_state = [tuple(
            seq_decoder_residual_block_init(
                enc, size=size, rate=rate_i, in_dim=latent_dim * 2,
                block_type=block_type
            )
            for rate_i in rate
        ) for i in range(num_blocks)]

        return init_state


def seq_bytenet_decoder(state_tm1, dec,
                        num_blocks=3, size=3,
                        rate=[1, 2, 4, 8, 16],
                        act='relu',
                        normalization='ln',
                        block_type='bytenet',
                        name=None, reuse=None):
    assert len(state_tm1) == num_blocks

    # loop dilated causal conv block
    with tf.variable_scope(name, "bytenet-decoder", values=[dec, state_tm1],
                           reuse=reuse):
        state_t = []

        for i, state_li_tm1 in zip(range(num_blocks), state_tm1):
            with tf.name_scope(f'bytenet-decoder-depth-{i}', values=[dec]):
                state_li_t = []

                for rate_i, state_li_tm1_di in zip(rate, state_li_tm1):
                    state_li_t_di, dec = seq_decoder_residual_block(
                        dec, state_li_tm1_di, size=size, rate=rate_i,
                        act=act,
                        normalization=normalization,
                        block_type=block_type,
                        name=f'decoder-res-block.{i}.{size}.{rate_i}'
                    )
                    state_li_t.append(state_li_t_di)

                # save state for next iteration
                state_t.append(tuple(state_li_t))

        return state_t, dec
