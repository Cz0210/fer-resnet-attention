#!/bin/bash
#SBATCH -J ppt_visuals
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/ppt-visuals-%j.out
#SBATCH --error=logs/ppt-visuals-%j.err

set -eo pipefail
source ~/.bashrc
conda activate fer_cv
set -u

PROJECT_DIR=/share/home/u20526/czx/CV
cd "${PROJECT_DIR}"
export PYTHONPATH=$PWD
mkdir -p logs assets/figures/ppt

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
echo "====================="

if [[ ! -f outputs/resnet18/predictions.csv ]]; then
  echo "Missing outputs/resnet18/predictions.csv. Re-run src.evaluate first." >&2
  exit 1
fi
if [[ ! -d face_images/test ]]; then
  echo "Missing face_images/test." >&2
  exit 1
fi
if [[ ! -f outputs/resnet18/best.pt ]]; then
  echo "Missing outputs/resnet18/best.pt." >&2
  exit 1
fi

python scripts/make_real_image_results.py \
  --predictions outputs/resnet18/predictions.csv \
  --data_dir face_images \
  --save_dir assets/figures/ppt \
  --n 12

python -m src.gradcam_vis \
  --config configs/resnet18.yaml \
  --checkpoint outputs/resnet18/best.pt \
  --data_dir face_images \
  --output_dir assets/figures/ppt \
  --num_images 16

find assets/figures/ppt -type f | sort
