#!/bin/bash
#SBATCH --job-name=qwen_n64_dance
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --time=18:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/diss_dance/logs/qwen/qwen_n64_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/diss_dance/scripts/qwen/all_64.py
