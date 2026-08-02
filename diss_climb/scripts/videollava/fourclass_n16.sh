#!/bin/bash
#SBATCH --job-name=FV_n16
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/dissertation_v2/logs/videollava/fourclass_n16_%j.log
 
set -e
source ~/dissertation/venv/bin/activate
nvidia-smi
 
python3 ~/dissertation/repo/dissertation_v2/scripts/videollava/fourclass_n16.py

