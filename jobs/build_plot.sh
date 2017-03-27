#!/bin/sh
#BSUB -q gpuqueuetitanx
#BSUB -J build-plot
#BSUB -n 1
#BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -W 10:00
#BSUB -u amwebdk@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o log-%J.out
#BSUB -e log-%J.err

export BASE_SAVE_DIR=/work1/$USER/kandidat
export PYTHONPATH=./:./sugartensor/
source ~/stdpy3/bin/activate

python3 code/plot/semi-supervised-synthetic-digits-grid.py
