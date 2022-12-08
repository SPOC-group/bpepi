#!/bin/bash

for seed in $(seq 10)
do
	#for snap_time in 0 3 5 6 7 8 9 10 11 12 14 17 20 25 30
	for delta in 0.0001 0.0003 0.0006 0.0008 0.001 0.0012 0.0015 0.002 0.0025 0.003
	do
		echo $seed
		echo $delta
		sed "s/seed=.*/seed=$seed/" run_oc_SI_snap2.sh > temp0.sh
		sleep 0.01
		sed "s/delta=.*/delta=$delta/"  temp0.sh > temp.sh
    	sleep 0.01
    	sbatch temp.sh
	done
done
