#!/bin/bash

for seed in $(seq 20)
do
	#for snap_time in 0 5 8 10 12 14 16 18 20 22 24 26 28 30 35 40
	for snap_time in 15 17
	do
		echo $seed
		echo $snap_time
		sed "s/seed=.*/seed=$seed/" run_oc_SI_snap.sh > temp0.sh
		sed "s/snap_time=.*/snap_time=$snap_time/"  temp0.sh > temp.sh
    		sleep 0.01
    		sbatch temp.sh &
	done
done
