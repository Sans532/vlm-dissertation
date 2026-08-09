#!/bin/bash
#SBATCH --job-name=bestexo_R
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --time=05:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/diss_climb/logs/qwen/qwen_reasoning_bestexo_%j.log

source ~/dissertation/venv/bin/activate
python3 ~/dissertation/repo/diss_climb/scripts/qwen_reasoning_bestexo.py
