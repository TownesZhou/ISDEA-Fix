#!/usr/local/bin/bash
#SBATCH --job-name=submit
#SBATCH --output=submit.stdout.txt
#SBATCH --error=submit.stderr.txt
#SBATCH --partition=ramos-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#
set -e

#
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPI_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

#
task=NELL9951
rm -rf datasets/${task}Trans/processed/*
rm -rf datasets/${task}Ind/processed/*
rm -rf experiments/NBFNet/Ind${task}Trans/*
rm -rf experiments/NBFNet/Ind${task}Ind/*
rm -rf experiments/NBFNet/Ind${task}PermInd/*
for seed in 42; do
    #
    python -u script/run.py -c config/${task}-trans.yaml --gpus [0] --myid ${seed} --resume null
    python -u script/run.py -c config/${task}-ind.yaml --gpus [0] --myid ${seed} --resume "$(pwd)/experiments/NBFNet/Ind${task}Trans/${seed}"
    python -u script/run.py -c config/${task}-perm-ind.yaml --gpus [0] --myid ${seed} --resume "$(pwd)/experiments/NBFNet/Ind${task}Trans/${seed}"
done
