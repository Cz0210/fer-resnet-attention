#!/bin/bash
#SBATCH -J fer_eval_all
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

DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/face_images}
EVAL_SPLIT=${EVAL_SPLIT:-test}

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
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing data dir: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -f "src/evaluate.py" ]]; then
  echo "Missing evaluation entry: ${PROJECT_DIR}/src/evaluate.py" >&2
  exit 1
fi
if [[ ! -f "src/visualize_results.py" ]]; then
  echo "Missing visualization entry: ${PROJECT_DIR}/src/visualize_results.py" >&2
  exit 1
fi
if [[ ! -f "scripts/collect_results.py" ]]; then
  echo "Missing collection script: ${PROJECT_DIR}/scripts/collect_results.py" >&2
  exit 1
fi

mkdir -p assets/figures

declare -a RUNS=(
  "cnn_baseline:configs/cnn_baseline.yaml"
  "resnet18:configs/resnet18.yaml"
  "resnet18_cbam:configs/resnet18_cbam.yaml"
  "resnet18_cbam_focal:configs/resnet18_cbam_focal.yaml"
)

for item in "${RUNS[@]}"; do
  run_name="${item%%:*}"
  config_path="${PROJECT_DIR}/${item#*:}"
  checkpoint="${PROJECT_DIR}/outputs/${run_name}/best.pt"
  out_dir="${PROJECT_DIR}/outputs/${run_name}"

  if [[ ! -f "${config_path}" ]]; then
    echo "Skip ${run_name}: missing config ${config_path}" >&2
    continue
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Skip ${run_name}: missing checkpoint ${checkpoint}"
    continue
  fi

  echo "===== EVALUATE ${run_name} ====="
  python -m src.evaluate \
    --checkpoint "${checkpoint}" \
    --config "${config_path}" \
    --data_dir "${DATA_DIR}" \
    --split "${EVAL_SPLIT}" \
    --output_dir "${out_dir}"

  python -m src.visualize_results \
    --run_dir "${out_dir}" \
    --assets_dir "${PROJECT_DIR}/assets/figures" \
    --outputs_dir "${PROJECT_DIR}/outputs"
done

python -m scripts.collect_results \
  --outputs_dir "${PROJECT_DIR}/outputs" \
  --assets_dir "${PROJECT_DIR}/assets/figures"

echo "Evaluation and result collection finished."
