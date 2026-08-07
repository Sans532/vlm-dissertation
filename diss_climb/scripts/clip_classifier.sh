#!/bin/bash
#SBATCH --job-name=clip_classifier
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/dissertation/repo/diss_climb/logs/clip_classifier_%j.log

source ~/dissertation/venv/bin/activate
python3 ~/dissertation/repo/diss_climb/scripts/clip_mitigation_classifier.py
