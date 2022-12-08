#!/bin/bash
#SBATCH --job-name=BPEpi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.log
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=ghio.1749811@studenti.uniroma1.it

python="/home/ghio/miniconda3/envs/sib/bin/python"
script="/home/ghio/bpepi/Analysis/sim_oc_dSIR.py"
save_dir="./data/SI_path/"
save_DF_dir="./data_frames/SI_path/"
graph="path"
N=10000
d=2
lam=0.8
delta=0.1
#rho=0.4
rho="0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1."
nsim=1
seed=33

$python $script --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --delta $delta --rho $rho --nsim $nsim --seed $seed --sens --SI
