# External data layout

Datasets are not redistributed in this repository. After obtaining each dataset according to its license and the paper's preprocessing protocol, place the generated task-stream files and splits under this directory.

The task-stream manifests are included under `configs/`. Supply the external splits in this layout:

```text
data/
└── datasets/
    ├── musique/{train,validation,anchor,test}.jsonl
    ├── proofwriter/{train,validation,anchor,test}.jsonl
    ├── gsm8k/{train,validation,anchor,test}.jsonl
    ├── hotpotqa/{train,validation,anchor,test}.jsonl
    ├── coco_detection/{train,validation,test}.jsonl
    ├── coco_caption/{train,validation,test}.jsonl
    ├── refcoco_grounding/{train,validation,test}.jsonl
    └── vqav2/{train,validation,test}.jsonl
```

Prepare disjoint splits according to the paper protocol. The main run configurations apply per-task limits of 250 adaptation examples, 50 validation examples, and 500 test examples.

## Reproducible split builders

### Multimodal stream

The recovered COCO/RefCOCO/VQAv2 builder and task prompts are provided in
`scripts/build_coco_hcl_tasks.py` and `scripts/coco_hcl_rows.py`. They use only
the Python standard library. Prepare COCO 2014 images in `train2014/` and
`val2014/` under the image root, and supply these annotations:

```text
annotations/
├── instances_{train,val}2014.json
├── captions_{train,val}2014.json
├── v2_OpenEnded_mscoco_{train,val}2014_questions.json
├── v2_mscoco_{train,val}2014_annotations.json
└── refcoco/{instances.json,refs(unc).p}
```

From the repository root, generate a new 250/50/500 stream:

```bash
python scripts/build_coco_hcl_tasks.py \
  --instruction-dir /path/to/annotations \
  --image-root /path/to/coco/images \
  --output-dir data/coco_hcl_250_50_500 \
  --train-per-task 250 --validation-per-task 50 --test-per-task 500 \
  --seed 630
```

Point `task_stream_path` in the selected run configuration to
`data/coco_hcl_250_50_500/taskstream_coco_hcl.json`. The generated manifest uses
relative split paths; image paths are resolved to absolute paths. Use a new
output directory because existing split files are overwritten. `--download`
can fetch VQAv2 and RefCOCO annotations; COCO images and detection/caption
annotations must already be present. Only load trusted RefCOCO pickle files.

Sampling consumes one `random.Random(630)` in Detection, Caption, Grounding,
VQAv2 order. Detection/Caption/VQAv2 use train2014 for adaptation and val2014
for evaluation. RefCOCO uses train references for adaptation and pools
val/testA/testB/test references for validation/test, taking the first expression
per reference. Each split selects at most one record per image. The recovered
sampler prefers unblocked images but falls back to the available pool when it
is too small; missing image files are excluded before sampling.

`split_manifest.json` records the **actual** selected record IDs, image IDs,
split-file checksums, seed, and within-task image intersections. Check these
intersections before running an experiment; any overlap is printed as a warning.
Annotation order/version and available images affect selection. The original
main-run sample-ID manifest has not been recovered, so this builder does not
guarantee the exact historical paper split. The recovered sampling and prompt
logic is retained; CLI defaults are now explicitly 250/50/500 (the archived
script used 500/100/100), and the manifest is new audit output.

The `anchor` entry in the generated task manifest aliases validation, matching
the recovered manifest. Main HCL profiles instead collect anchors from selected
training batches; consult `memory.anchor_selection_mode` and
`memory.anchor_count_per_batch` in the run configuration.

### Textual stream

The recovered textual-reasoning stream builder is integrated as
`scripts/build_textual_stream.py`. It preserves the experiment seed and the
task-specific stratified sampling protocol. The default output is `data/datasets/`:

```bash
python scripts/build_textual_stream.py \
  --source-root /path/to/existing/gsm8k-proofwriter-hotpotqa/splits \
  --musique-root /path/to/musique/data
```

For a task-stream manifest whose train files have not yet been split, use the
generic deterministic JSON/JSONL splitter. It resolves dataset paths the same
way as the runtime loader and does not overwrite files unless requested:

```bash
python scripts/split_task_stream.py \
  --config configs/my_taskstream.json \
  --output-config configs/my_taskstream_split.json \
  --val-ratio 0.2
```

The recovered multimodal 500/100/500 manifest is available at
`configs/taskstream_multimodal_recovered_500_100_500.json`. Place its JSON split files under
`data/datasets_multimodal_500_100_500/`. Validation is explicitly reused as the
anchor split, matching the recovered experiment directory. Image paths inside
the external records may need relocation for the local COCO installation.

The exact sampled record files for the independent textual budget sweep were recovered from the experiment server. They are not redistributed here because benchmark data remain subject to their original licenses. Place them under `data/datasets_budget_300_80_80_600/` using the layout in `configs/taskstream_textual_budget_300_80_80_600.json`.

The recovered budget-sweep counts are 300 train, 80 validation, 80 anchor, and 600 test records for every task. For GSM8K, ProofWriter, and HotpotQA, the recovered anchor file is byte-identical to the validation file; MuSiQue uses separate validation and anchor samples. This is recorded explicitly rather than silently claiming all evaluator subsets are mutually disjoint. Test records remain separate.

The additional server also contains larger ordered source pools for the main textual and multimodal runs. Because no dedicated 250/50/500 manifest was present, the exact main-run sampled record IDs remain unconfirmed in this release instead of being inferred from those larger pools.

The included single-task textual stream expects:

```text
data/
└── datasets/
    └── musique/
        ├── train.jsonl
        ├── validation.jsonl
        ├── test.jsonl
        └── anchor.jsonl
```

Task-stream JSON files can point to any project-relative layout. Avoid adding personal absolute paths to committed configurations.
