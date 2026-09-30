from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260718


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the four-task textual-reasoning HCL data splits."
    )
    parser.add_argument(
        "--musique-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "musique_data" / "data",
        help="Directory containing musique_ans_v1.0_{train,dev}.jsonl.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "source_4task",
        help="Directory containing existing gsm8k/proofwriter/hotpotqa split folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "datasets",
        help="Destination used by configs/taskstream_textual_main_250_50_500.json.",
    )
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--anchor-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sizes = {
        "train": args.train_size,
        "validation": args.validation_size,
        "anchor": args.anchor_size,
        "test": args.test_size,
    }
    if any(size < 0 for size in sizes.values()):
        raise ValueError("Split sizes must be non-negative.")

    for task_name in ("gsm8k", "proofwriter", "hotpotqa"):
        build_existing_task(args.source_root, args.output_root, task_name, sizes=sizes)
    build_musique(
        args.musique_root,
        args.output_root / "musique",
        sizes=sizes,
        seed=args.seed,
    )

    print(f"Prepared four-task data at {args.output_root.resolve()}")
    for task_name in ("gsm8k", "proofwriter", "hotpotqa", "musique"):
        counts = {
            path.stem: count_jsonl(path)
            for path in sorted((args.output_root / task_name).glob("*.jsonl"))
        }
        print(f"{task_name}: {counts}")
    return 0


def build_existing_task(
    source_root: Path,
    output_root: Path,
    task_name: str,
    *,
    sizes: dict[str, int],
) -> None:
    source_dir = source_root / task_name
    splits = {
        split: read_jsonl(source_dir / f"{split}.jsonl")[: sizes[split]]
        for split in ("train", "validation", "test")
    }
    for split, rows in splits.items():
        require_count(f"{task_name} {split}", rows, sizes[split])

    # This matches the recovered experiment protocol for the three existing tasks.
    # A distinct MuSiQue anchor sample is constructed below.
    require_count(
        f"{task_name} validation-as-anchor",
        splits["validation"],
        sizes["anchor"],
    )
    splits["anchor"] = [dict(row) for row in splits["validation"][: sizes["anchor"]]]
    write_splits(output_root / task_name, splits)


def build_musique(
    source_root: Path,
    output_dir: Path,
    *,
    sizes: dict[str, int],
    seed: int,
) -> None:
    train_source = read_jsonl(source_root / "musique_ans_v1.0_train.jsonl")
    dev_source = read_jsonl(source_root / "musique_ans_v1.0_dev.jsonl")
    train_pool = stratified_shuffle(train_source, key=musique_hop_count, seed=seed + 300)
    required = sizes["train"] + sizes["validation"] + sizes["anchor"]
    require_count("musique train/validation/anchor", train_pool, required)
    raw_splits = {
        "train": train_pool[: sizes["train"]],
        "validation": train_pool[
            sizes["train"] : sizes["train"] + sizes["validation"]
        ],
        "anchor": train_pool[
            sizes["train"] + sizes["validation"] : required
        ],
        "test": stratified_sample(
            dev_source,
            size=sizes["test"],
            key=musique_hop_count,
            seed=seed + 301,
        ),
    }
    splits = {
        split: [convert_musique(row, split, index) for index, row in enumerate(rows)]
        for split, rows in raw_splits.items()
    }
    assert_disjoint_source_ids(splits)
    write_splits(output_dir, splits)


def convert_musique(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    answer = str(row.get("answer", "")).strip()
    if not answer:
        raise ValueError(f"MuSiQue row has no answer: {row.get('id')}")
    context = [
        {
            "idx": paragraph.get("idx"),
            "title": str(paragraph.get("title", "")),
            "paragraph_text": str(paragraph.get("paragraph_text", "")),
        }
        for paragraph in row.get("paragraphs", [])
    ]
    if not context:
        raise ValueError(f"MuSiQue row has no paragraphs: {row.get('id')}")
    return {
        "id": f"musique_{split}_{index:05d}",
        "question": (
            f"{str(row.get('question', '')).strip()}\n\n"
            "Answer using the provided context. Return only the shortest final answer; do not explain."
        ),
        "visible_context": context,
        "answer": answer,
        "answers": unique_nonempty([answer, *row.get("answer_aliases", [])]),
        "task_name": "musique",
        "task_type": "compositional_multi_hop_qa",
        "metadata": {
            "source": "musique",
            "original_id": row.get("id"),
            "hop_count": musique_hop_count(row),
            "paragraph_count": len(context),
        },
    }


def musique_hop_count(row: dict[str, Any]) -> int:
    identifier = str(row.get("id", ""))
    try:
        return int(identifier.split("hop", 1)[0])
    except ValueError:
        decomposition = row.get("question_decomposition", [])
        return len(decomposition) if isinstance(decomposition, list) else 0


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    size: int,
    key: Callable[[dict[str, Any]], Hashable],
    seed: int,
) -> list[dict[str, Any]]:
    if size > len(rows):
        raise ValueError(f"Cannot sample {size} rows from a population of {len(rows)}")
    grouped = _shuffled_groups(rows, key=key, seed=seed)
    exact = {name: size * len(group) / len(rows) for name, group in grouped.items()}
    allocation = {name: int(value) for name, value in exact.items()}
    remaining = size - sum(allocation.values())
    remainder_order = sorted(
        grouped,
        key=lambda name: (exact[name] - allocation[name], str(name)),
        reverse=True,
    )
    for name in remainder_order[:remaining]:
        allocation[name] += 1
    selected = [
        row
        for name in sorted(grouped, key=str)
        for row in grouped[name][: allocation[name]]
    ]
    random.Random(seed).shuffle(selected)
    return selected


def stratified_shuffle(
    rows: list[dict[str, Any]],
    *,
    key: Callable[[dict[str, Any]], Hashable],
    seed: int,
) -> list[dict[str, Any]]:
    """Interleave shuffled strata so contiguous slices retain every stratum."""
    grouped = _shuffled_groups(rows, key=key, seed=seed)
    consumed = {name: 0 for name in grouped}
    output: list[dict[str, Any]] = []
    while len(output) < len(rows):
        candidates = [name for name in grouped if consumed[name] < len(grouped[name])]
        name = max(
            candidates,
            key=lambda candidate: (
                len(grouped[candidate]) / len(rows)
                - consumed[candidate] / max(len(output), 1),
                -consumed[candidate],
                str(candidate),
            ),
        )
        output.append(grouped[name][consumed[name]])
        consumed[name] += 1
    return output


def _shuffled_groups(
    rows: list[dict[str, Any]],
    *,
    key: Callable[[dict[str, Any]], Hashable],
    seed: int,
) -> dict[Hashable, list[dict[str, Any]]]:
    grouped: dict[Hashable, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
    return grouped


def assert_disjoint_source_ids(splits: dict[str, list[dict[str, Any]]]) -> None:
    seen: dict[Any, str] = {}
    for split, rows in splits.items():
        for row in rows:
            identifier = row["metadata"]["original_id"]
            if identifier in seen:
                raise ValueError(
                    f"MuSiQue source id {identifier!r} appears in both {seen[identifier]} and {split}"
                )
            seen[identifier] = split


def unique_nonempty(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_splits(output_dir: Path, splits: dict[str, list[dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_count(label: str, rows: list[Any], expected: int) -> None:
    if len(rows) < expected:
        raise ValueError(f"{label} has {len(rows)} rows, but {expected} are required")


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


if __name__ == "__main__":
    raise SystemExit(main())
