#!/bin/bash
#SBATCH --job-name=BPEpi_snap2
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run_snap9A.log
#SBATCH --mem=32GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ghio.1749811@studenti.uniroma1.it

python="/home/ghio/miniconda3/envs/sib/bin/python"
script="/home/ghio/bpepi/Analysis/sim_oc_dSIR.py"
save_dir="./data/SI_Dsnap_lam7_scale/"
save_DF_dir="./data_frames/SI_Dsnap_lam7_scale/"
graph="rrg"
N=80000
d=3
lam=0.7
#n_sources=1
delta=0.003
rho=1.
nsim=1
seed=10
snap_time=15
damping=0.2
#is=10
n_iter=200
it_max=20000

$python $script --damping $damping --n_iter $n_iter --it_max $it_max --snap_time $snap_time --save_dir $save_dir --save_DF_dir $save_DF_dir --graph $graph --N $N --d $d --lam $lam --delta $delta --rho $rho --nsim $nsim --seed $seed --snap --SI --rnd_inf_init 
