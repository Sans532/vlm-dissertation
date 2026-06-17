#!/bin/bash
#SBATCH --job-name=qwen_benchmark
#SBATCH --partition=Teaching
#SBATCH --nodelist=landonia11
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/logs/benchmark_%j.log

source ~/dissertation/venv/bin/activate
python3 ~/dissertation/repo/scripts/run_benchmark.py
