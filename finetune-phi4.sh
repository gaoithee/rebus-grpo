#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="phi4"
#SBATCH --partition=Main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=500G
#SBATCH --output=slurm_outputs/finetune-phi4-mini.out
#SBATCH --cpus-per-task=8

# Standard preamble for debugging
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"

source ~/.bashrc

conda init bash
conda activate rebus-env
accelerate launch finetune-phi4.py


echo "DONE!"
