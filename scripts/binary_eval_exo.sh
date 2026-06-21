#!/bin/bash
#SBATCH --job-name=eval_binary
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/logs/eval_binary_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/scripts/binary_eval.py
