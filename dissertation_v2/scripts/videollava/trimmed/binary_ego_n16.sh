#!/bin/bash
#SBATCH --job-name=bV16Tego
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=03:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/videollava/trimmed/binary_ego16_trimmed_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/dissertation_v2/scripts/videollava/trimmed/binary_ego_n16.py
