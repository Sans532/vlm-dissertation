#!/bin/bash
#SBATCH --job-name=quick_test
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/logs/quick_test_%j.log

source ~/dissertation/venv/bin/activate
python3 ~/dissertation/repo/scripts/quick_test.py
