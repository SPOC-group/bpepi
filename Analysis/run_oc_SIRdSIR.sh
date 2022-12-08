#!/bin/bash
#SBATCH --job-name=BPEpi
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.log
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=ghio.1749811@studenti.uniroma1.it

python="/home/ghio/miniconda3/envs/sib/bin/python"
script="/home/ghio/bpepi/Analysis/sim_oc_dSIR.py"
save_dir="./data/SIRdSIR/"
save_DF_dir="./data_frames/SIRdSIR/"
graph="rrg"
N=10000
d=3
lam=0.8
delta=0.1
rho=0.4
#rho="0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1."
nsim=1
seed=33
mu=0.5
is=10

$python $script --save_marginals --print_it --iter_space $is --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --delta $delta --rho $rho --nsim $nsim --seed $seed --sens --mu $mu
