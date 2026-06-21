#!/bin/bash
#SBATCH --job-name=eval_egoexo
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/logs/eval_egoexo_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/scripts/binary_eval_egoexo.py
