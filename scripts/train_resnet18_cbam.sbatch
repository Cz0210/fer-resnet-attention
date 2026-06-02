#!/bin/bash
#SBATCH --job-name=fer_r18_cbam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# Edit these paths for your HPC cluster, or override them when submitting:
#   PROJECT_DIR=/your/project/path DATA_DIR=/your/data/path sbatch scripts/train_resnet18_cbam.sbatch
PROJECT_DIR="${PROJECT_DIR:-/share/home/u20526/czx/fer-resnet-attention}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/face_images}"
CONDA_ENV="${CONDA_ENV:-fer_cv}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONFIG_PATH="configs/resnet18_cbam.yaml"
RUN_NAME="resnet18_cbam"

cd "${PROJECT_DIR}"
mkdir -p logs outputs assets/figures

echo "========== Slurm Job Info =========="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Hostname: $(hostname)"
echo "PWD: $(pwd)"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'not available')"
echo "Project dir: ${PROJECT_DIR}"
echo "Data dir: ${DATA_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "===================================="

if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
else
  echo "Cannot find conda activation script: ${CONDA_SH}" >&2
  echo "Set CONDA_SH=/path/to/conda.sh before sbatch if your cluster uses another path." >&2
  exit 1
fi

conda activate "${CONDA_ENV}"

echo "Python path: $(which python)"
echo "Active conda env: ${CONDA_DEFAULT_ENV:-unknown}"
python - <<'PY'
import torch

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
PY

echo "========== Training resnet18_cbam =========="
python -m src.train \
  --config "${CONFIG_PATH}" \
  --data_dir "${DATA_DIR}" \
  --output_dir outputs

echo "Training finished. Outputs: outputs/${RUN_NAME}"
