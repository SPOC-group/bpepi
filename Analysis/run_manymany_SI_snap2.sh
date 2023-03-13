#!/bin/bash
N=100000
for seed in $(seq 1 10)
	do
	for snap_time in 0 1 2 3 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10 11 12 13
		do
			for delta in 0.005 0.01 0.02 0.03 0.04 0.05
			#for snap_time in 15 17
			do
				echo $delta
				echo $snap_time
				echo $seed
				sed "s/seed=.*/seed=$seed/" run_oc_SI_snap_torch.sh > temp0.sh
				sleep 0.02
				sed "s/snap_time=.*/snap_time=$snap_time/"  temp0.sh > temp1.sh
				sleep 0.02
				sed "s/delta=.*/delta=$delta/"  temp1.sh > temp2.sh
				sleep 0.02
				sed "s/npippo/n$N/"  temp2.sh > temp3.sh
				sleep 0.02
				sed "s/N=.*/N=$N/"  temp3.sh > temp4.sh
				sleep 0.02
				sbatch temp4.sh
		done
	done
done
