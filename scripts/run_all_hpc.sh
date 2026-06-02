#!/bin/bash
set -euo pipefail

# Submit all four training jobs in parallel. Run from the project root:
#   bash scripts/run_all_hpc.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

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
  result="$(sbatch "${script}")"
  echo "${name}: ${result}"
  if [[ "${result}" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
    echo "${name} job id: ${BASH_REMATCH[1]}"
  fi
done
