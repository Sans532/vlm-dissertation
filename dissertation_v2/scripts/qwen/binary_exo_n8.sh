#!/bin/bash
#SBATCH --job-name=B_exo
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/qwen/binary_exo_n8_%j.log
 
set -e
source ~/dissertation/venv/bin/activate
nvidia-smi
 
python3 ~/dissertation/repo/dissertation_v2/scripts/qwen/binary_exo_n8.py
