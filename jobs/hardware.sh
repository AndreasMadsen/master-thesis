#!/bin/sh
#BSUB -q gpuqueuetitanx
#BSUB -J hardware
#BSUB -n 1
#BSUB -R "rusage[ngpus_excl_p=4]"
#BSUB -W 00:30
#BSUB -u amwebdk@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o log-%J.out
#BSUB -e log-%J.err

echo "GPU INFO"
nvidia-smi

echo "CPU INFO"
cat /proc/cpuinfo

echo "RAM INFO"
cat /proc/meminfo
