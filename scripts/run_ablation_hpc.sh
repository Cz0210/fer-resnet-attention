#!/bin/bash
set -eo pipefail
set -u

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/CV}
cd "${PROJECT_DIR}"
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
  if [[ ! -f "${config}" ]]; then
    echo "Missing ablation config: ${config}" >&2
    exit 1
  fi
  result="$(sbatch \
    --job-name="fer_${run_name}" \
    --export=ALL,CONFIG_PATH="${config}",RUN_NAME="${run_name}" \
    scripts/train_ablation.sbatch)"
  echo "${name}: ${result}"
  if [[ "${result}" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
    echo "${name} job id: ${BASH_REMATCH[1]}"
  fi
done
