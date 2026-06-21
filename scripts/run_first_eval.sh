#!/bin/bash
#SBATCH --job-name=first_eval
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=08:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/logs/first_eval_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/scripts/run_first_eval.py
