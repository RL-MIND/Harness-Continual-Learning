"""Build the recovered multimodal stream; see data/README.md for provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

if __package__:
    from .coco_hcl_rows import caption_rows, detection_rows, load_coco, read_json, refcoco_rows, vqa_rows, write_json
else:
    from coco_hcl_rows import caption_rows, detection_rows, load_coco, read_json, refcoco_rows, vqa_rows, write_json


VQA_FILES = {
    "v2_OpenEnded_mscoco_train2014_questions.json": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "v2_mscoco_train2014_annotations.json": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "v2_OpenEnded_mscoco_val2014_questions.json": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "v2_mscoco_val2014_annotations.json": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}

REFCOCO_URLS = (
    "https://web.archive.org/web/20220413011718id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
    "http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HCL task-stream JSON files from COCO-style datasets.")
    parser.add_argument("--instruction-dir", default="instruction")
    parser.add_argument("--image-root", required=True, help="COCO images directory with train2014/ and val2014/.")
    parser.add_argument("--output-dir", default="data/coco_hcl_250_50_500")
    parser.add_argument("--download", action="store_true", help="Download VQAv2 and RefCOCO annotations if missing.")
    parser.add_argument("--train-per-task", type=int, default=250)
    parser.add_argument("--validation-per-task", type=int, default=50)
    parser.add_argument("--test-per-task", type=int, default=500)
    parser.add_argument("--seed", type=int, default=630)
    args = parser.parse_args()
    if min(args.train_per_task, args.validation_per_task, args.test_per_task) < 0:
        parser.error("Split counts must be non-negative.")

    instruction_dir = Path(args.instruction_dir)
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir)
    if args.download:
        download_missing_annotations(instruction_dir)

    rng = random.Random(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    used_train_image_ids: set[int] = set()

    tasks = [
        build_detection_task(instruction_dir, image_root, output_dir, rng, used_train_image_ids, args),
        build_caption_task(instruction_dir, image_root, output_dir, rng, used_train_image_ids, args),
        build_refcoco_task(instruction_dir, image_root, output_dir, rng, used_train_image_ids, args),
        build_vqa_task(instruction_dir, image_root, output_dir, rng, used_train_image_ids, args),
    ]
    stream = {
        "stream_name": "coco_hcl_stream",
        "description": "COCO-based HCL stream: detection, captioning, RefCOCO grounding, and VQAv2.",
        "tasks": tasks,
    }
    stream_path = output_dir / "taskstream_coco_hcl.json"
    write_json(stream_path, stream)
    write_split_manifest(output_dir, tasks, args.seed)
    print(json.dumps({"task_stream": str(stream_path), "tasks": [task["task_name"] for task in tasks]}, indent=2))
    return 0


def download_missing_annotations(instruction_dir: Path) -> None:
    instruction_dir.mkdir(parents=True, exist_ok=True)
    for expected, url in VQA_FILES.items():
        if (instruction_dir / expected).exists():
            continue
        download_zip_member(url, expected, instruction_dir)
    ref_dir = instruction_dir / "refcoco"
    if not ((ref_dir / "instances.json").exists() and (ref_dir / "refs(unc).p").exists()):
        for url in REFCOCO_URLS:
            try:
                download_and_extract_zip(url, instruction_dir)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Unable to download RefCOCO annotations: {last_error}")


def download_zip_member(url: str, expected_name: str, output_dir: Path) -> None:
    archive_path = output_dir / Path(url).name
    download_file(url, archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        member = next((name for name in zf.namelist() if Path(name).name == expected_name), None)
        if member is None:
            raise FileNotFoundError(f"{expected_name} not found in {archive_path}")
        zf.extract(member, output_dir)
        extracted = output_dir / member
        target = output_dir / expected_name
        if extracted != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), str(target))


def download_and_extract_zip(url: str, output_dir: Path) -> None:
    archive_path = output_dir / Path(url).name
    if archive_path.suffix != ".zip":
        archive_path = output_dir / "refcoco.zip"
    download_file(url, archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(output_dir)


def download_file(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as f:
        shutil.copyfileobj(response, f)


def build_detection_task(
    instruction_dir: Path,
    image_root: Path,
    output_dir: Path,
    rng: random.Random,
    used_train_image_ids: set[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    train = load_coco(instruction_dir / "instances_train2014.json")
    val = load_coco(instruction_dir / "instances_val2014.json")
    categories = {int(item["id"]): item["name"] for item in train["categories"]}
    train_rows = detection_rows(train, image_root / "train2014", categories, split="train")
    val_rows = detection_rows(val, image_root / "val2014", categories, split="val")
    splits = write_task_splits(
        output_dir / "coco_detection",
        train_rows,
        val_rows,
        rng,
        used_train_image_ids,
        args,
    )
    return task_spec("coco_detection", "coco_detection", splits)


def build_caption_task(
    instruction_dir: Path,
    image_root: Path,
    output_dir: Path,
    rng: random.Random,
    used_train_image_ids: set[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_rows = caption_rows(load_coco(instruction_dir / "captions_train2014.json"), image_root / "train2014", split="train")
    val_rows = caption_rows(load_coco(instruction_dir / "captions_val2014.json"), image_root / "val2014", split="val")
    splits = write_task_splits(output_dir / "coco_caption", train_rows, val_rows, rng, used_train_image_ids, args)
    return task_spec("coco_caption", "coco_caption", splits)


def build_refcoco_task(
    instruction_dir: Path,
    image_root: Path,
    output_dir: Path,
    rng: random.Random,
    used_train_image_ids: set[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ref_dir = instruction_dir / "refcoco"
    rows = refcoco_rows(ref_dir / "instances.json", ref_dir / "refs(unc).p", image_root / "train2014")
    train_rows = [row for row in rows if row["metadata"].get("ref_split") == "train"]
    val_rows = [row for row in rows if row["metadata"].get("ref_split") in {"val", "testA", "testB", "test"}]
    splits = write_task_splits(output_dir / "refcoco_grounding", train_rows, val_rows, rng, used_train_image_ids, args)
    return task_spec("refcoco_grounding", "refcoco_grounding", splits)


def build_vqa_task(
    instruction_dir: Path,
    image_root: Path,
    output_dir: Path,
    rng: random.Random,
    used_train_image_ids: set[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_rows = vqa_rows(
        instruction_dir / "v2_OpenEnded_mscoco_train2014_questions.json",
        instruction_dir / "v2_mscoco_train2014_annotations.json",
        image_root / "train2014",
        split="train",
    )
    val_rows = vqa_rows(
        instruction_dir / "v2_OpenEnded_mscoco_val2014_questions.json",
        instruction_dir / "v2_mscoco_val2014_annotations.json",
        image_root / "val2014",
        split="val",
    )
    splits = write_task_splits(output_dir / "vqav2", train_rows, val_rows, rng, used_train_image_ids, args)
    return task_spec("vqav2", "vqa", splits)


def write_task_splits(
    task_dir: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    rng: random.Random,
    used_train_image_ids: set[int],
    args: argparse.Namespace,
) -> dict[str, str]:
    task_dir.mkdir(parents=True, exist_ok=True)
    train = sample_rows(train_rows, args.train_per_task, rng, avoid_image_ids=used_train_image_ids)
    used_train_image_ids.update(int(row["metadata"]["image_id"]) for row in train)
    blocked = {int(row["metadata"]["image_id"]) for row in train}
    validation = sample_rows(val_rows, args.validation_per_task, rng, avoid_image_ids=blocked)
    blocked.update(int(row["metadata"]["image_id"]) for row in validation)
    test = sample_rows(val_rows, args.test_per_task, rng, avoid_image_ids=blocked)
    paths = {
        "train": task_dir / "train.json",
        "validation": task_dir / "validation.json",
        "test": task_dir / "test.json",
        "anchor": task_dir / "validation.json",
    }
    write_json(paths["train"], train)
    write_json(paths["validation"], validation)
    write_json(paths["test"], test)
    # Task-stream split paths are resolved relative to taskstream_coco_hcl.json.
    # Keeping them relative makes generated streams self-contained when this
    # project directory is moved or copied to another machine.
    return {key: f"{task_dir.name}/{path.name}" for key, path in paths.items()}


def sample_rows(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    *,
    avoid_image_ids: set[int],
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    available = [row for row in rows if Path(str(row.get("image", ""))).exists()]
    preferred = [row for row in available if int(row["metadata"]["image_id"]) not in avoid_image_ids]
    pool = preferred if len(preferred) >= count else available
    shuffled = list(pool)
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    seen_images: set[int] = set()
    for row in shuffled:
        image_id = int(row["metadata"]["image_id"])
        if image_id in seen_images:
            continue
        selected.append(row)
        seen_images.add(image_id)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} unique-image rows available; requested {count}.")
    return selected


def task_spec(task_name: str, task_type: str, splits: dict[str, str]) -> dict[str, Any]:
    return {"task_name": task_name, "task_type": task_type, "modality": ["image", "text"], "interactions": 1, "splits": splits}


def write_split_manifest(output_dir: Path, tasks: list[dict[str, Any]], seed: int) -> None:
    """Record actual selected IDs and overlaps, rather than inferring old splits."""
    manifest: dict[str, Any] = {"seed": seed, "task_order": [], "tasks": {}}
    for task in tasks:
        name = task["task_name"]
        manifest["task_order"].append(name)
        splits = {}
        image_sets = {}
        for split in ("train", "validation", "test"):
            path = output_dir / task["splits"][split]
            rows = read_json(path)
            image_sets[split] = {int(row["metadata"]["image_id"]) for row in rows}
            splits[split] = {
                "count": len(rows),
                "record_ids": [row["id"] for row in rows],
                "image_ids": [row["metadata"]["image_id"] for row in rows],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        overlaps = {
            f"{left}/{right}": sorted(image_sets[left] & image_sets[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        }
        manifest["tasks"][name] = {"splits": splits, "image_id_intersections": overlaps}
        if any(overlaps.values()):
            print(f"WARNING: {name} has image overlap; inspect split_manifest.json before use.")
    write_json(output_dir / "split_manifest.json", manifest)


if __name__ == "__main__":
    raise SystemExit(main())
