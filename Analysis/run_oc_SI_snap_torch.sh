#!/bin/bash
#SBATCH --job-name=bpepi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.log
#SBATCH --mem=4GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=indaco.biazzo@epfl.ch

python="/home/biazzo/miniconda3/envs/sib/bin/python"
script="/home/biazzo/git/bpepi/Analysis/sim_oc_dSIR.py"
save_dir="./data/snap_indaco/npippo/"
save_DF_dir="./data_frames/snap_df_indaco/npippo/"
graph="rrg"
N=2000
d=3
lam=0.6
#n_sources=1
delta=0.09
rho=1.
nsim=1
seed=1
snap_time=1
damping=0.2
#is=10
n_iter=200
it_max=20000
pytorch=1 # Using pytorch backend

$python $script --save_marginals --damping $damping --n_iter $n_iter --it_max $it_max --snap_time $snap_time --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --delta $delta --rho $rho --nsim $nsim --pytorch $pytorch --seed $seed --snap --SI --rnd_inf_init 
