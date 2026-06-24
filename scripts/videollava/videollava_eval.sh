#!/bin/bash
#SBATCH --job-name=videollava
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/logs/videollava_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/scripts/videollava/videollava_eval.py
