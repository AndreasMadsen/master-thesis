
.PHONY: test report report-auto report-spellcheck report-clean

export PYTHONPATH=./code
XELATEX=xelatex -file-line-error -interaction=nonstopmode

test:
	nosetests --nologcapture -v -s \
		code/tf_operator/decoder_residual_block/test/*.py \
		code/tf_operator/encoder_residual_block/test/*.py \
		code/tf_operator/seq_dense/test/*.py \
		code/model/test/*.py
		code/dataset/test/*.py

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
