#!/bin/sh
#BSUB -q gpuqueuetitanx
#BSUB -J bytenet-small-selu-wmt-bilingual-news-timeing
#BSUB -n 6
#BSUB -R "rusage[ngpus_excl_p=4]"
#BSUB -W 24:00
#BSUB -u amwebdk@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o log-%J.out
#BSUB -e log-%J.err

# export TF_USE_XLA=1
export BASE_SAVE_DIR=/work1/$USER/kandidat
export PYTHONPATH=./:./sugartensor/
source ~/stdpy3/bin/activate

python3 code/script/bytenet_small_selu_wmt_bilingual_news_timeing.py
