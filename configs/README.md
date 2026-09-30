# Experiment configuration inventory

This directory contains the textual-reasoning and multimodal configurations retained for reproducibility. Interactive ALFWorld and Minecraft configurations live under the repository's `experiments/` directory. No metrics, predictions, logs, checkpoints, figures, or other prior-run outputs are included.

## Path convention

- `configs/taskstream_textual_main_250_50_500.json`: included portable manifest for MuSiQue, ProofWriter, GSM8K, and HotpotQA.
- `configs/taskstream_multimodal_main_250_50_500.json`: included portable manifest for COCO detection, COCO captioning, RefCOCO grounding, and VQAv2.
- `data/datasets/`: externally supplied dataset splits.
- `storage/experiments/<configuration-name>/`: generated logs, caches, checkpoints, manifests, and predictions. `storage/` is ignored by Git.

Relative paths are resolved from the repository root by the CLI configuration loader.

## HCL reasoning experiments

- `deepseek_flash_reasoning_hcl_plasticity_250_50_500.json`
- `deepseek_flash_reasoning_hcl_stability_250_50_500.json`
- `qwen36_27b_reasoning_hcl_stability_capabilities_250_50_500.json`

## Independent textual historical-loss-budget sweep

- `deepseek_v4_flash_textual_budget_b0_300_80_80_600.json`
- `deepseek_v4_flash_textual_budget_b1_300_80_80_600.json`
- `deepseek_v4_flash_textual_budget_b3_300_80_80_600.json`
- `deepseek_v4_flash_textual_budget_binf_300_80_80_600.json`
- `taskstream_textual_budget_300_80_80_600.json`

These four runnable profiles use frozen DeepSeek-V4-Flash, the paper's 300 adaptation / 80 validation / 80 historical-anchor / 600 test allocation, and ten 30-example update batches per task stage. The profiles are identical in experimental settings except for `optimizer.historical_loss_budget`, which is 0, 1, 3, or `null` ($B_n=\infty$). The optimizer generates three alternatives inside each update opportunity, matching the recovered experiment configuration; that internal multiplicity is distinct from the paper's count of ten proposal opportunities per task stage.

## Distillation comparison experiments

- `run_stability_hcl_01_4b.json` (`Base-HCL`): Qwen3-4B performs final answering, online structuring and selection, optimizer candidate generation, and memory operations.
- `run_stability_hcl_02_4b_opt27b.json` (`Opt-HCL`): Qwen3.6-27B generates optimizer candidates; Qwen3-4B retains all online and memory roles.
- `run_stability_hcl_03_4b_teacher27b.json` (`Online-HCL`): Qwen3.6-27B performs online task structuring, workflow and artifact selection, and memory operations; Qwen3-4B retains final answering and optimizer candidate generation.
- `taskstream_textual_main_250_50_500.json`: the portable MuSiQue, ProofWriter, GSM8K, and HotpotQA task stream shared by all three runs.

The three profiles share the same Stability-HCL gate, three optimizable components, candidate budget, task stream, data allocation, memory limits, and update schedule. With a batch size of 300 and 250 adaptation examples per task, each task stage has one optimizer update opportunity. The `selection_model` and `memory_model` fields make the Online-HCL role split explicit; when absent, both roles fall back to the main model.

The local checkpoints are expected at `models/Qwen3-4B/` and `models/Qwen3.6-27B/`. These directories are not included. Paths can be changed to checkpoints supplied by the evaluation environment without changing the experimental method or hyperparameters.

## Reasoning baselines

- `deepseek_flash_reasoning_raw_zero_shot_500.json`

## Multimodal experiments and baselines

- `qwen36_27b_coco_gpu1_direct_raw_zero_shot_500.json`
- `qwen36_27b_coco_gpu1_v0_harness_500.json`
- `qwen36_27b_coco_gpu1_hcl_plasticity_250_50_500.json`
- `qwen36_27b_coco_gpu1_hcl_stability_250_50_500.json`

## Task-stream configurations

- `taskstream_textual_main_250_50_500.json`
- `taskstream_multimodal_main_250_50_500.json`
- `taskstream_textual_budget_300_80_80_600.json`

Configuration filenames containing a GPU number preserve the original experiment-to-device scheduling label. They contain no hostname, username, institution, filesystem location, or hardware identifier unique to the authors, and do not require that same device number when reproduced.

The previous component-ablation matrix is intentionally not included as a paper configuration: it used Qwen3.6-27B and disabled runtime components, whereas the paper specifies Qwen3.5-4B and freezes only persistent updates while keeping each component available. A search of the additional experiment server found no Qwen3.5-4B four-way persistent-update-freeze configuration, so this remains unresolved rather than being approximated.

The stronger-model comparison is the Stability-HCL series above. Earlier `run_paper_*` profiles represented a separate four-run study and are intentionally excluded to avoid conflating that study with the paper's Base/Opt/Online comparison.
