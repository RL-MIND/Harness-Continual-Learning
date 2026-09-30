#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-configs/run_stability_hcl_01_4b.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

python main.py --config "${config_path}" --splits train,test "$@"
