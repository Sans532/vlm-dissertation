#!/bin/bash
#SBATCH --job-name=video_10
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=02:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/qwen/video10_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi

# Check ffmpeg is available
which ffmpeg || echo "WARNING: ffmpeg not found, trimming will be skipped"

python3 ~/dissertation/repo/dissertation_v2/scripts/qwen/video_10.py
