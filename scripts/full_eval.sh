#!/bin/bash
#SBATCH --job-name=full_eval
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/logs/full_eval_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/scripts/full_eval.py
