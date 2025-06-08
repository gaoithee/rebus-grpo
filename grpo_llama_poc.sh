#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="grpo"
#SBATCH --partition=Main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --output=slurm_outputs/llama-grpo.out
#SBATCH --cpus-per-task=8

# Standard preamble for debugging
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"

source ~/.bashrc

conda init bash
conda activate vllm_env
accelerate launch grpo_llama_poc.py


echo "DONE!"
