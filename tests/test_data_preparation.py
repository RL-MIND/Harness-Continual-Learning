from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from hcl.dataset import DatasetLoader, load_task_stream_specs


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DataPreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_script("build_textual_stream")
        cls.splitter = load_script("split_task_stream")

    def test_split_records_is_deterministic_and_disjoint(self) -> None:
        records = [{"id": index} for index in range(20)]
        first = self.splitter.split_records(
            records, val_ratio=0.2, min_val=1, max_val=None, seed=42
        )
        second = self.splitter.split_records(
            records, val_ratio=0.2, min_val=1, max_val=None, seed=42
        )
        self.assertEqual(first, second)
        train, validation = first
        self.assertEqual(len(train), 16)
        self.assertEqual(len(validation), 4)
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in validation})

    def test_textual_builder_creates_distinct_train_validation_anchor_test(self) -> None:
        def source_row(hop: int, index: int) -> dict[str, object]:
            return {
                "id": f"{hop}hop__{index}",
                "question": f"question {index}",
                "answer": f"answer {index}",
                "answer_aliases": [f"answer {index}", f"alias {index}"],
                "paragraphs": [
                    {"idx": 0, "title": "title", "paragraph_text": "context"}
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            train_rows = [source_row(2 + index % 3, index) for index in range(12)]
            dev_rows = [source_row(2 + index % 3, 100 + index) for index in range(6)]
            for filename, rows in (
                ("musique_ans_v1.0_train.jsonl", train_rows),
                ("musique_ans_v1.0_dev.jsonl", dev_rows),
            ):
                (source / filename).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            output = root / "output"
            self.builder.build_musique(
                source,
                output,
                sizes={"train": 6, "validation": 3, "anchor": 3, "test": 6},
                seed=20260718,
            )
            split_ids = {}
            for split, expected in (
                ("train", 6),
                ("validation", 3),
                ("anchor", 3),
                ("test", 6),
            ):
                rows, shape = self.splitter.read_records(output / f"{split}.jsonl")
                self.assertEqual(shape, "jsonl")
                self.assertEqual(len(rows), expected)
                self.assertTrue(all(row["answers"][0] == row["answer"] for row in rows))
                split_ids[split] = {row["metadata"]["original_id"] for row in rows}
            names = list(split_ids)
            for index, name in enumerate(names):
                for other in names[index + 1 :]:
                    self.assertFalse(split_ids[name] & split_ids[other])

    def test_recovered_multimodal_manifest_is_runtime_compatible(self) -> None:
        manifest = ROOT / "configs" / "taskstream_multimodal_recovered_500_100_500.json"
        specs = load_task_stream_specs(manifest)
        self.assertEqual(
            [spec["task_name"] for spec in specs],
            ["coco_detection", "coco_caption", "refcoco_grounding", "vqav2"],
        )
        self.assertEqual(specs[-1]["task_type"], "vqa")
        for spec in specs:
            self.assertEqual(spec["splits"]["anchor"], spec["splits"]["validation"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(json.dumps([{"id": "one", "question": "q", "answer": "a"}]))
            loaded = DatasetLoader().load(path)
            self.assertEqual(loaded[0]["task_id"], "one")


if __name__ == "__main__":
    unittest.main()
