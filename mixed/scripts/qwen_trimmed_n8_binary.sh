#!/bin/bash
#SBATCH --job-name=Qb8
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=03:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/mixed/logs/qwen_trimmed_n8_binary_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/mixed/scripts/qwen_trimmed_n8_binary.py
