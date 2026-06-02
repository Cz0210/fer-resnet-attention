#!/bin/bash
set -eo pipefail
set -u

# Usage:
#   bash scripts/sync_results_from_hpc.sh user@hpc.example.edu [/remote/project/dir] [local/project/dir]

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/sync_results_from_hpc.sh user@host [remote_project_dir] [local_project_dir]" >&2
  exit 1
fi

HPC_HOST="$1"
REMOTE_PROJECT_DIR="${2:-/share/home/u20526/czx/CV}"
LOCAL_PROJECT_DIR="${3:-$(pwd)}"

mkdir -p "${LOCAL_PROJECT_DIR}/outputs" "${LOCAL_PROJECT_DIR}/logs" "${LOCAL_PROJECT_DIR}/assets/figures"

echo "Syncing from ${HPC_HOST}:${REMOTE_PROJECT_DIR}"
rsync -avz --progress "${HPC_HOST}:${REMOTE_PROJECT_DIR}/outputs/" "${LOCAL_PROJECT_DIR}/outputs/"
rsync -avz --progress "${HPC_HOST}:${REMOTE_PROJECT_DIR}/logs/" "${LOCAL_PROJECT_DIR}/logs/"
rsync -avz --progress "${HPC_HOST}:${REMOTE_PROJECT_DIR}/assets/figures/" "${LOCAL_PROJECT_DIR}/assets/figures/"
echo "Sync finished."
