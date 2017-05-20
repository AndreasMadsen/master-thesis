#!/bin/sh
#BSUB -q gpuqueuetitanx
#BSUB -J bytenet2-europarl
#BSUB -n 6
#BSUB -R "rusage[ngpus_excl_p=4]"
#BSUB -W 30:00
#BSUB -u amwebdk@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o log-%J.out
#BSUB -e log-%J.err

export BASE_SAVE_DIR=/work1/$USER/kandidat
export PYTHONPATH=./:./sugartensor/
source ~/stdpy3/bin/activate

python3 code/script/bytenet2_europarl_train.py
