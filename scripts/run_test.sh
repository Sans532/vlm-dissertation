#!/bin/bash
#SBATCH --job-name=qwen_test
#SBATCH --partition=Teaching
#SBATCH --nodelist=landonia11
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/logs/test_%j.log

source ~/dissertation/venv/bin/activate
python3 ~/dissertation/repo/scripts/test_qwen.py
