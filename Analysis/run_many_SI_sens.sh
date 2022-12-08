#!/bin/bash

for seed in $(seq 100)
#for seed in 26 27 28 29 30 31 32 33 34 35 36
do
	echo $seed
	sed "s/seed=.*/seed=$seed/" run_oc_SI_sens.sh > temp.sh
    	sleep 0.1
    	sbatch temp.sh &
done
