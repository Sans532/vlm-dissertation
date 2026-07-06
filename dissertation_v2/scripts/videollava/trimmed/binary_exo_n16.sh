#!/bin/bash
#SBATCH --job-name=bV16Texo
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/videollava/trimmed/binary_exo16_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/dissertation_v2/scripts/videollava/trimmed/binary_exo_n16.py
