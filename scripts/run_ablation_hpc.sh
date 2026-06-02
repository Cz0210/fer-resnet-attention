#!/bin/bash
set -euo pipefail

# Submit five ablation experiments in parallel. Run from the project root:
#   bash scripts/run_ablation_hpc.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

declare -a JOBS=(
  "No Augmentation:configs/ablation/resnet18_no_aug.yaml:ablation_resnet18_no_aug"
  "Augmentation:configs/ablation/resnet18_aug.yaml:ablation_resnet18_aug"
  "CBAM + Augmentation:configs/ablation/resnet18_cbam_aug.yaml:ablation_resnet18_cbam_aug"
  "CBAM + Augmentation + Focal:configs/ablation/resnet18_cbam_aug_focal.yaml:ablation_resnet18_cbam_aug_focal"
  "CBAM + Augmentation + Focal + Sampler:configs/ablation/resnet18_cbam_aug_focal_sampler.yaml:ablation_resnet18_cbam_aug_focal_sampler"
)

for item in "${JOBS[@]}"; do
  name="${item%%:*}"
  rest="${item#*:}"
  config="${rest%%:*}"
  run_name="${rest#*:}"
  result="$(sbatch \
    --job-name="fer_${run_name}" \
    --export=ALL,CONFIG_PATH="${config}",RUN_NAME="${run_name}" \
    scripts/train_ablation.sbatch)"
  echo "${name}: ${result}"
  if [[ "${result}" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
    echo "${name} job id: ${BASH_REMATCH[1]}"
  fi
done
