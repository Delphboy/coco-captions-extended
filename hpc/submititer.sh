#!/bin/bash

declare lengths=(100)

for len in "${lengths[@]}"
do
                echo "Running for length $len"
                qsub -v LENGTH=$len -N train_len=$len train.qsub
                qsub -v LENGTH=$len -N restval=$len restval.qsub
                qsub -v LENGTH=$len -N val=$len val.qsub
                qsub -v LENGTH=$len -N test=$len test.qsub
done

sleep 5
qstat
