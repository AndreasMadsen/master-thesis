#!/bin/sh
#PBS -N bytenet-nltk-comtrans
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -m eba
#PBS -M amwebdk@gmail.com
#PBS -q k40_interactive

cd $PBS_O_WORKDIR

# Enable python3
export PYTHONPATH=./
source ~/stdpy3/bin/activate

CUDA_VISIBLE_DEVICES=3 python3 code/script/bytenet_nltk_comtrans_train.py
