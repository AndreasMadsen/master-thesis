#!/bin/sh
#BSUB -q gpuqueuetitanx
#BSUB -J bytenet-nltk-comtrans
#BSUB -n 4
#BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -W 6:00
#BSUB -u amwebdk@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o log-%J.out
#BSUB -e log-%J.err

export TF_USE_XLA=1
export PYTHONPATH=./
source ~/stdpy3/bin/activate

python3 code/script/bytenet_nltk_comtrans_train.py
