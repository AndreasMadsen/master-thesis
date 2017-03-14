#!/bin/sh
#BSUB -q gpuqueuetitanx
#BSUB -J build-europarl
#BSUB -n 1
#BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -W 05:00
#BSUB -u amwebdk@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o log-%J.out
#BSUB -e log-%J.err

export PYTHONPATH=./
source ~/stdpy3/bin/activate

python3 code/script/build_europarl.py
