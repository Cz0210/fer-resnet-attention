#!/bin/bash
set -eo pipefail
set -u

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/CV}
cd "${PROJECT_DIR}"
mkdir -p logs

declare -a JOBS=(
  "CNN baseline:scripts/train_cnn_baseline.sbatch"
  "ResNet18:scripts/train_resnet18.sbatch"
  "ResNet18-CBAM:scripts/train_resnet18_cbam.sbatch"
  "ResNet18-CBAM-Focal:scripts/train_resnet18_cbam_focal.sbatch"
)

for item in "${JOBS[@]}"; do
  name="${item%%:*}"
  script="${item#*:}"
  if [[ ! -f "${script}" ]]; then
    echo "Missing sbatch script: ${script}" >&2
    exit 1
  fi
  result="$(sbatch "${script}")"
  echo "${name}: ${result}"
  if [[ "${result}" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
    echo "${name} job id: ${BASH_REMATCH[1]}"
  fi
done
