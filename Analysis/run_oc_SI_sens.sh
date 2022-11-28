#!/bin/bash
#SBATCH --job-name=BPEpi_sens
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.log
#SBATCH --mem=32GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ghio.1749811@studenti.uniroma1.it

python="/home/ghio/miniconda3/envs/sib/bin/python"
script="/home/ghio/bpepi/Analysis/sim_oc_dSIR.py"
save_dir="./data/SI_Dsens/"
save_DF_dir="./data_frames/SI_Dsens/"
graph="rrg"
N=10000
d=3
lam=1.
n_sources=1
#delta="0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99"
rho=0.
nsim=1
seed=33
damping=0.25
#is=10
n_iter=500
it_max=50000

$python $script --damping $damping --n_iter $n_iter --it_max $it_max --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --n_sources $n_sources --rho $rho --nsim $nsim --seed $seed --sens --SI --rnd_inf_init 
