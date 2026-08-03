#!/bin/bash
#SBATCH --job-name=q3Cf
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --time=03:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/diss_climb/logs/qwen3vl/qwen3vl_fourclass_climbing_%j.log

source ~/dissertation/venv/bin/activate
nvidia-smi
python3 ~/dissertation/repo/diss_climb/scripts/qwen3vl/fourclass.py
