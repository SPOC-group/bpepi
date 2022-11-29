#!/bin/bash

for seed in $(seq 10)
do
	#for snap_time in 0 3 5 6 7 8 9 10 11 12 14 17 20 25 30
	for snap_time in 0 2 5 6 7 8 9 10 11 12 13 14 15 16 18 20 23 27 30
	#for snap_time in 15 17
	do
		echo $seed
		echo $snap_time
		sed "s/seed=.*/seed=$seed/" run_oc_SI_snap2.sh > temp0.sh
		sed "s/snap_time=.*/snap_time=$snap_time/"  temp0.sh > temp.sh
    		sleep 0.01
    		sbatch temp.sh &
	done
done
