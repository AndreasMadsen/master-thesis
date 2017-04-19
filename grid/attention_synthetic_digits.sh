#!/bin/sh

for TRAIN in 64 128 256
do
  TRAIN_SIZE=$TRAIN bsub < jobs/attention_synthetic_digits.sh
done
