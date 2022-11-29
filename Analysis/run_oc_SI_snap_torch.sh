#!/bin/bash
#SBATCH --job-name=bpepi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=run.log
#SBATCH --mem=8GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=indaco.biazzo@epfl.ch

python="/home/biazzo/miniconda3/envs/sib/bin/python"
script="/home/biazzo/git/bpepi/Analysis/sim_oc_dSIR.py"
save_dir="./data/snap_indaco/"
save_DF_dir="./data_frames/snap_df_indaco/"
graph="rrg"
N=10
d=3
lam=0.6
#n_sources=1
delta=0.1
rho=1.
nsim=1
seed=33
snap_time=7.3
damping=0.2
#is=10
n_iter=500
it_max=500
pytorch=0 # Using pytorch backend

$python $script --save_marginals --damping $damping --n_iter $n_iter --it_max $it_max --snap_time $snap_time --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --delta $delta --rho $rho --nsim $nsim --pytorch $pytorch --seed $seed --snap --SI --rnd_inf_init 
