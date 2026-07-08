#!/bin/bash
#SBATCH --job-name=qwen_3cls
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/qwen/3class_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/dissertation_v2/scripts/qwen/3class.py
