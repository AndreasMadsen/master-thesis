#!/bin/sh
#BSUB -q gpuqueuetitanx
#BSUB -J semi-bytenet-wmt-bilingual-news
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

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/appl/cuda/8.0/extras/CUPTI/lib64/
python3 code/script/semi_bytenet_wmt_bilingual_news_train.py
