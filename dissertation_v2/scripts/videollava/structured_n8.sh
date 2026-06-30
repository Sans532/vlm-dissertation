#!/bin/bash
#SBATCH --job-name=SV_n8
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200:1
#SBATCH --nodelist=saxa
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/videollava/structured_n8_%j.log
 
set -e
source ~/dissertation/venv/bin/activate
nvidia-smi
 
python3 ~/dissertation/repo/dissertation_v2/scripts/videollava/structured_n8.py

