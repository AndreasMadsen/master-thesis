#!/bin/sh

for TRAIN in 64 128 256
do
  for SEMI in 0 512 1024
  do
    for FACTOR in 0.2 0.1 0.01
    do
      for ITERATION in 1 2 3 4 5
      do
        TRAIN_SIZE=$TRAIN SEMI_SIZE=$SEMI SEMI_FACTOR=$FACTOR ITERATION=$ITERATION bsub < jobs/semi_bytenet_synthetic_digits.sh
      done
    done
  done
done
