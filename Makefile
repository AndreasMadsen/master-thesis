
export PYTHONPATH=./code

test:
	nosetests --nologcapture -v -s \
		code/tf_operator/decoder_residual_block/test/*.py \
		code/tf_operator/encoder_residual_block/test/*.py \
		code/tf_operator/seq_dense/test/*.py
