#!/usr/bin/env bash

while getopts "l:r:t:" opt; do
  case $opt in
    l)
      REF_LANG=$OPTARG
      ;;
    r)
      REF_FILE=$OPTARG
      ;;
    t)
      TRANSLATE_FILE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# alias the required perl scripts
alias multi-bleu='perl code/deps/mosesdecoder/scripts/generic/multi-bleu.perl'
alias tokenizer='perl code/deps/mosesdecoder/scripts/tokenizer/tokenizer.perl'
shopt -s expand_aliases

WORK_DIR=`mktemp -d -t blue-score`

# tokenize files
tokenizer -l $REF_LANG < $REF_FILE > $WORK_DIR/tokenized.target.txt 2> /dev/null
tokenizer -l $REF_LANG < $TRANSLATE_FILE > $WORK_DIR/tokenized.translate.txt 2> /dev/null

# compute "multi-bleu" bleu score
multi-bleu $WORK_DIR/tokenized.target.txt < $WORK_DIR/tokenized.translate.txt

# cleanup workdir
rm -rf $WORK_DIR
