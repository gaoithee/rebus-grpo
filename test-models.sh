#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="llama-test"
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1g.20gb
#SBATCH --time=48:00:00
#SBATCH --mem=500GB
#SBATCH --output=slurm_outputs/test-llama-coldstart-grpo.out
#SBATCH --cpus-per-task=8

# Standard preamble for debugging
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"

# conda activate rebus-env
accelerate launch test-llama.py


echo "DONE!"
