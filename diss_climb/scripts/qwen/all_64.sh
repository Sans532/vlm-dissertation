#!/bin/bash
#SBATCH --job-name=qwen_n64
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=18:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/diss_climb/logs/qwen_n64_%j.log

source ~/dissertation/venv/bin/activate
python3 ~/dissertation/repo/diss_climb/scripts/qwen/all_64.py
