#!/bin/bash
#SBATCH --job-name=eval_structured
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/logs/eval_structured_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/scripts/structured_eval.py
