
.PHONY: test report report-auto report-spellcheck report-clean

export PYTHONPATH=./
XELATEX=xelatex -file-line-error -interaction=nonstopmode

# --with-process-isolation is a plugin from "nosepipe"
test:
	nosetests --nologcapture --with-process-isolation -v -s \
		code/tf_operator/batch_repeat/test/*.py \
		code/tf_operator/bytenet_encoder/test/*.py \
		code/tf_operator/bytenet_decoder/test/*.py \
		code/tf_operator/decoder_residual_block/test/*.py \
		code/tf_operator/encoder_residual_block/test/*.py \
		code/tf_operator/select_value/test/*.py \
		code/tf_operator/seq_dense/test/*.py \
		code/tf_operator/seq_prop/test/*.py \
		code/dataset/test/*.py \
		code/model/test/*.py

report: report/thesis.tex
	cd report && latexmk -pdf -pdflatex="$(XELATEX)" -use-make thesis.tex

report-auto: report/thesis.tex
	cd report && latexmk -pdf -pdflatex="$(XELATEX)" -use-make -pvc thesis.tex

report-spellcheck:
	find report/ -name "*.tex" -exec aspell --lang=en --mode=tex --dont-backup check "{}" \;
	aspell --lang=da --mode=tex --dont-backup check report/frontmatter/summary-danish.tex

report-clean:
	rm -f report/thesis.pdf report/*.aux
	rm -f report/**/*.aux
