#!/bin/bash

for seed in $(seq 35 110)
do
	echo $seed
	sed "s/seed=.*/seed=$seed/" run_oc_SI_snap.sh > temp.sh
    	sleep 0.1
    	sbatch temp.sh &
done
