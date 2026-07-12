#!/bin/bash
#SBATCH --job-name=fQ16T
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --time=03:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/diss_dance/logs/qwen/trimmed/fourclass_n16_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/diss_dance/scripts/qwen/trimmed/fourclass_n16.py
