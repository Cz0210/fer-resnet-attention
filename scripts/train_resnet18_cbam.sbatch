#!/bin/bash
#SBATCH -J fer_r18_cbam
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
source ~/.bashrc
conda activate fer_cv
set -u

PROJECT_DIR=/share/home/u20526/czx/CV
cd "${PROJECT_DIR}"
export PYTHONPATH=$PWD
mkdir -p logs outputs

RUN_NAME=resnet18_cbam
CONFIG_PATH=${PROJECT_DIR}/configs/resnet18_cbam.yaml
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/face_images}
OUT_DIR=${PROJECT_DIR}/outputs/${RUN_NAME}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.001}
SEED=${SEED:-42}

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
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
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing data dir: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -f "src/train.py" ]]; then
  echo "Missing training entry: ${PROJECT_DIR}/src/train.py" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "===== TRAIN ====="
echo "run name: ${RUN_NAME}"
echo "config: ${CONFIG_PATH}"
echo "data dir: ${DATA_DIR}"
echo "output dir: ${OUT_DIR}"
echo "epochs: ${EPOCHS}"
echo "batch size: ${BATCH_SIZE}"
echo "lr: ${LR}"
echo "seed: ${SEED}"

python -m src.train \
  --config "${CONFIG_PATH}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${PROJECT_DIR}/outputs" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --seed "${SEED}"
