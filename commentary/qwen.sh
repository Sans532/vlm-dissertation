#!/bin/bash
#SBATCH --job-name=qwen_comm
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/commentary/qwen_comm_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/commentary/qwen.py
