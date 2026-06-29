#!/bin/bash
#SBATCH --job-name=bin_V_n32
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200:1 
#SBATCH --time=04:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/videollava/binary_n32_%j.log
 
set -e
source ~/dissertation/venv/bin/activate
nvidia-smi

python3 ~/dissertation/repo/dissertation_v2/scripts/videollava/binary_n32.py

