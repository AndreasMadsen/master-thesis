#!/bin/sh
#BSUB -q gpuqueuetitanx
#BSUB -J semi-bytenet-synthetic-digits
#BSUB -n 6
#BSUB -R "rusage[ngpus_excl_p=4]"
#BSUB -W 02:00
#BSUB -u amwebdk@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o log-%J.out
#BSUB -e log-%J.err

export PYTHONPATH=./
source ~/stdpy3/bin/activate

python3 code/script/semi_bytenet_synthetic_digits_train.py
