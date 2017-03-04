#!/bin/sh
#BSUB -q gpuqueuetitanx
#BSUB -J bytenet-europarl
#BSUB -n 6
#BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -W 39:00
#BSUB -u amwebdk@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o log-%J.out
#BSUB -e log-%J.err

export PYTHONPATH=./
source ~/stdpy3/bin/activate

python3 code/script/bytenet_europarl.py
