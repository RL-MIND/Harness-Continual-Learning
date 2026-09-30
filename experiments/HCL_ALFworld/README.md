# HCL ALFWorld experiments

This module contains the ALFWorld experiment implementation, configurations, task sequences, and command-line entry point used with the unified HCL submission.

## Installation

Python 3.9 through 3.12 is supported by the package metadata.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[llm]'
```

Use the `local` extra instead of `llm` when running a local Transformers model, or install both extras when both backends are needed.

## Usage

```bash
hcl-alfworld --help
```

Experiment definitions are under `configs/`, and the supplied domain-incremental task sequences are under `sequences/`. External ALFWorld assets and model credentials are intentionally not bundled. API credentials should be supplied through the environment variable named by the selected configuration's `api_key_env` field.

The `scripts/bootstrap.sh` helper documents the environment bootstrap expected by this module.
