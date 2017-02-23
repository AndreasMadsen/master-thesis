#!/bin/sh
#BSUB -q gpuqueuetitanx
#BSUB -J semi-bytenet-nltk-comtrans
#BSUB -n 6
#BSUB -R "rusage[ngpus_excl_p=2]"
#BSUB -W 12:00
#BSUB -u amwebdk@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o log-%J.out
#BSUB -e log-%J.err

export PYTHONPATH=./
source ~/stdpy3/bin/activate

python3 code/script/semi_bytenet_nltk_comtrans_train.py
