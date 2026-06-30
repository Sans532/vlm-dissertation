#!/bin/bash
#SBATCH --job-name=jigsaws_Q
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/logs/jigsaw_qwen_binary_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/scripts/jigsaw_qwen.py
