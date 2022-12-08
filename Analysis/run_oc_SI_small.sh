#!/bin/bash
#SBATCH --job-name=BPEpi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.log
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghio.1749811@studenti.uniroma1.it

python="/home/ghio/miniconda3/envs/sib/bin/python"
script="/home/ghio/bpepi/Analysis/sim_oc_dSIR.py"
save_dir="./data/SI_small/"
save_DF_dir="./data_frames/SI_tree/"
graph="tree"
N=10000
d=3
lam=1.
delta=0.0005
#delta="0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99"
rho="0. 0.002 0.004 0.006 0.008"
nsim=1
seed=33

$python $script --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --delta $delta --rho $rho --nsim $nsim --seed $seed --sens --SI 
