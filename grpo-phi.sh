#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="phi"
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_outputs/grpo-phi.out
#SBATCH --cpus-per-task=8

# Standard preamble for debugging
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"

# conda activate rebus-env

accelerate launch grpo-phi.py


echo "DONE!"
