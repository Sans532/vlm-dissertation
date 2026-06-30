#!/bin/bash
#SBATCH --job-name=video_10
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/qwen/video10_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/dissertation_v2/scripts/qwen/video_10.py
