#!/bin/bash
#SBATCH --job-name=BPEpi_snap2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.log
#SBATCH --mem=32GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ghio.1749811@studenti.uniroma1.it

python="/home/ghio/miniconda3/envs/sib/bin/python"
script="/home/ghio/bpepi/Analysis/sim_oc_dSIR.py"
save_dir="./data/SI_Dsnap_delta/"
save_DF_dir="./data_frames/SI_Dsnap_delta/"
graph="rrg"
N=10000
d=3
lam=0.6
#n_sources=1
delta=0.001
rho=1.
nsim=1
seed=33
snap_time=50
damping=0.2
#is=10
n_iter=500
it_max=50000
pytorch=0 # Using pytorch backend

$python $script --damping $damping --n_iter $n_iter --it_max $it_max --snap_time $snap_time --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --delta $delta --rho $rho --nsim $nsim --seed $seed --pytorch $pytorch --snap --SI --rnd_inf_init 
