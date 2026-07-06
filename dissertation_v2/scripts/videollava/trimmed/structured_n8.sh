#!/bin/bash
#SBATCH --job-name=sV8T
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=03:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/videollava/trimmed/structured_n8_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/dissertation_v2/scripts/videollava/trimmed/structured_n8.py
