#!/bin/bash
#SBATCH --job-name=BPEpi
#SBATCH --time=24:00:00
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
save_dir="./data/SI_scale/"
save_DF_dir="./data_frames/SI_scale/"
graph="rrg"
N=256000
d=3
lam=0.8
delta=0.01
rho=0.05
nsim=1
seed=3

$python $script --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --delta $delta --rho $rho --nsim $nsim --seed $seed --sens --SI 
