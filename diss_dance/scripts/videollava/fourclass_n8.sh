#!/bin/bash
#SBATCH --job-name=fV8E
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --time=03:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/diss_dance/logs/videollava/fourclass_n8%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/diss_dance/scripts/videollava/fourclass_n8.py
