#!/bin/bash
#SBATCH --job-name=BPEpi_snap
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.log
#SBATCH --mem=32GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ghio.1749811@studenti.uniroma1.it

python="/home/ghio/miniconda3/envs/sib/bin/python"
script="/home/ghio/bpepi/Analysis/sim_oc_dSIR.py"
save_dir="./data/SI_Csnap2B/"
save_DF_dir="./data_frames/SI_Csnap2B/"
graph="rrg"
N=10000
d=3
lam=0.9
n_sources=1
#delta=0.02
rho=1.
nsim=1
seed=33
snap_time=50
#damping=0.4
#is=10
n_iter=2500
it_max=5000

$python $script --n_iter $n_iter --it_max $it_max --snap_time $snap_time --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --n_sources $n_sources --rho $rho --nsim $nsim --seed $seed --snap --SI --rnd_inf_init
