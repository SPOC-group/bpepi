#!/bin/bash

for seed in $(seq 25)
do
	echo $seed
	sed "s/seed=.*/seed=$seed/" run_oc_dSIR.sh > temp.sh
    	sleep 0.1
    	sbatch temp.sh &
done
