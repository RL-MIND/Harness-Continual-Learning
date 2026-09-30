#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${PROJECT_DIR}/.venv"
PKG_CACHE_DIR="${PROJECT_DIR}/.conda-pkgs"
DATA_DIR="${PROJECT_DIR}/data/alfworld"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  env CONDA_PKGS_DIRS="${PKG_CACHE_DIR}" XDG_CACHE_HOME="${PROJECT_DIR}/.cache" \
    conda create --prefix "${ENV_DIR}" python=3.9 pip -y
fi

env ALFWORLD_DATA="${DATA_DIR}" "${ENV_DIR}/bin/python" -m pip install --upgrade pip
env ALFWORLD_DATA="${DATA_DIR}" "${ENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}[dev,llm]"
if [[ ! -f "${DATA_DIR}/logic/alfred.pddl" ]] || \
   [[ ! -d "${DATA_DIR}/json_2.1.1/train" ]] || \
   [[ ! -f "${DATA_DIR}/detectors/mrcnn_alfred_objects_sep13_004.pth" ]]; then
  env ALFWORLD_DATA="${DATA_DIR}" "${ENV_DIR}/bin/alfworld-download" --data-dir "${DATA_DIR}"
fi
env ALFWORLD_DATA="${DATA_DIR}" "${ENV_DIR}/bin/hcl-alfworld" doctor --config "${PROJECT_DIR}/configs/hcl.yaml"
env ALFWORLD_DATA="${DATA_DIR}" "${ENV_DIR}/bin/hcl-alfworld" build-sequence --config "${PROJECT_DIR}/configs/hcl.yaml"

echo "Bootstrap complete. Activate with: conda activate ${ENV_DIR}"
