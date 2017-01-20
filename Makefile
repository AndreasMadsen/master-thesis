
export PYTHONPATH=./code

test:
	nosetests --nologcapture -v -s \
		code/tf_operator/decoder_residual_block/test/*.py
