#!/bin/bash
#SBATCH --job-name=diagnostic
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/qwen/diagnostic_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/dissertation_v2/scripts/qwen/diagnostic_test.py
