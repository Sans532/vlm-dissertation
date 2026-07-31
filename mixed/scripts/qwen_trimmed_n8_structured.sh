#!/bin/bash
#SBATCH --job-name=qwen_mixed_struct
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --time=05:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/mixed/logs/qwen_trimmed_n8_structured_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/mixed/scripts/qwen_trimmed_n8_structured.py
