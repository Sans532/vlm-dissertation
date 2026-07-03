#!/bin/bash
#SBATCH --job-name=D_IE
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=01:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/dissertation_v2/scripts/qwen/diagnostics/inter_expert.py > ~/dissertation/logs/ie_output.txt 2>&1
