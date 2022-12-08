#!/bin/bash

for seed in $(seq 10)
do
	for rho in 0.022 0.024 0.026 0.028 0.03 0.032 0.034 0.037 0.04 0.044 0.048 0.052
	#for rho in 0. 0.0005 0.001 0.0015 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 
	do
		echo $seed
		echo $rho
		sed "s/seed=.*/seed=$seed/" run_oc_SI_sens.sh > temp0.sh
		sleep 0.01
		sed "s/rho=.*/rho=$rho/"  temp0.sh > temp.sh
		sleep 0.01
		sbatch temp.sh &
	done
done
