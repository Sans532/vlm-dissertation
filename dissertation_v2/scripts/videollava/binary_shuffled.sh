#!/bin/bash
#SBATCH --job-name=bin_shuffled     
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/~/dissertation/repo/dissertation_v2/logs/videollava/binary_shuffled_%j.log
 
set -e
source ~/dissertation_v2/venv/bin/activate
nvidia-smi
 
python3 ~/dissertation/repo/dissertation_v2/scripts/videollava/binary_shuffled.py
