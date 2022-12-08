#!/bin/bash
#SBATCH --job-name=BPEpi
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.log
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghio.1749811@studenti.uniroma1.it

python="/home/ghio/miniconda3/envs/sib/bin/python"
script="/home/ghio/bpepi/Analysis/sim_oc_dSIR.py"
save_dir="./data/PDdSIR/"
save_DF_dir="./data_frames/PDdSIR/"
graph="rrg"
N=10000
d=3
lam=0.4
delta="0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4"
rho=1.
#rho="0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1."
nsim=1
seed=33
Delta=1

$python $script --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --delta $delta --rho $rho --nsim $nsim --seed $seed --sens --Delta $Delta
