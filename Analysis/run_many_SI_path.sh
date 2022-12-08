#!/bin/bash

for seed in $(seq 25)
do
	echo $seed
	sed "s/seed=.*/seed=$seed/" run_oc_SI_path.sh > temp.sh
    	sleep 0.3
    	sbatch temp.sh &
done
