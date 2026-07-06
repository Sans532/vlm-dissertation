#!/bin/bash
#SBATCH --job-name=RQ_n8
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/qwen/reasoning_n8_%j.log
 
set -e
source ~/dissertation/venv/bin/activate
nvidia-smi
 
python3 ~/dissertation/repo/dissertation_v2/scripts/qwen/reasoning_n8.py
