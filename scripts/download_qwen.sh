#!/bin/bash
#SBATCH --job-name=download_qwen
#SBATCH --partition=Teaching
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/%u/dissertation/logs/download_%j.log

source ~/dissertation/venv/bin/activate

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2-VL-7B-Instruct',
    local_dir='/home/$USER/dissertation/models/qwen2vl',
    ignore_patterns=['*.msgpack', '*.h5']
)
print('Qwen2-VL downloaded successfully.')
"
