from __future__ import annotations

import hashlib
import io
import json
import pickle
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hcl.dataset import DatasetLoader
from scripts import build_coco_hcl_tasks as builder


class MultimodalPreparationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.annotations = self.root / "annotations"
        self.annotations.mkdir()
        self.images = self.root / "images"
        self.output = self.root / "output"

    def prepare_sources(self) -> None:
        all_images = []
        all_annotations = []
        refs = []
        for split, offset in (("train", 0), ("val", 100)):
            image_dir = self.images / f"{split}2014"
            image_dir.mkdir(parents=True)
            images = []
            objects = []
            captions = []
            questions = []
            answers = []
            for index in range(12):
                image_id = offset + index
                filename = f"COCO_{split}2014_{image_id:012d}.jpg"
                (image_dir / filename).touch()
                images.append({"id": image_id, "file_name": filename, "width": 100, "height": 80})
                objects.append({
                    "id": image_id, "image_id": image_id, "category_id": 1,
                    "bbox": [10, 20, 30, 40], "iscrowd": 0,
                })
                captions.extend([
                    {"image_id": image_id, "caption": "a red bus"},
                    {"image_id": image_id, "caption": "a bus on a road"},
                ])
                questions.append({"question_id": image_id, "image_id": image_id, "question": "What color?"})
                answers.append({
                    "question_id": image_id,
                    "answers": [{"answer": "red"} for _ in range(10)],
                })
                refs.append({
                    "ref_id": image_id, "ann_id": image_id, "image_id": image_id,
                    "split": split, "sentences": [{"sent": "the red bus"}],
                })
                # RefCOCO image names are resolved under train2014 by the recovered builder.
                (self.images / "train2014" / filename).touch()
            categories = [{"id": 1, "name": "bus"}]
            builder.write_json(self.annotations / f"instances_{split}2014.json", {
                "images": images, "annotations": objects, "categories": categories,
            })
            builder.write_json(self.annotations / f"captions_{split}2014.json", {
                "images": images, "annotations": captions,
            })
            builder.write_json(self.annotations / f"v2_OpenEnded_mscoco_{split}2014_questions.json", {
                "questions": questions,
            })
            builder.write_json(self.annotations / f"v2_mscoco_{split}2014_annotations.json", {
                "annotations": answers,
            })
            all_images.extend(images)
            all_annotations.extend(objects)
        builder.write_json(self.annotations / "refcoco" / "instances.json", {
            "images": all_images, "annotations": all_annotations,
            "categories": [{"id": 1, "name": "bus"}],
        })
        with (self.annotations / "refcoco" / "refs(unc).p").open("wb") as handle:
            pickle.dump(refs, handle)

    def run_builder(self) -> None:
        arguments = [
            "build_coco_hcl_tasks.py", "--instruction-dir", str(self.annotations),
            "--image-root", str(self.images), "--output-dir", str(self.output),
            "--train-per-task", "2", "--validation-per-task", "2",
            "--test-per-task", "3", "--seed", "630",
        ]
        with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            self.assertEqual(builder.main(), 0)

    def test_full_builder_is_deterministic_loadable_and_records_actual_ids(self) -> None:
        self.prepare_sources()
        self.run_builder()
        manifest = builder.read_json(self.output / "split_manifest.json")
        expected_order = ["coco_detection", "coco_caption", "refcoco_grounding", "vqav2"]
        self.assertEqual(manifest["seed"], 630)
        self.assertEqual(manifest["task_order"], expected_order)
        for task_name in expected_order:
            task_manifest = manifest["tasks"][task_name]
            self.assertTrue(all(not overlap for overlap in task_manifest["image_id_intersections"].values()))
            for split, count in (("train", 2), ("validation", 2), ("test", 3)):
                path = self.output / task_name / f"{split}.json"
                rows = builder.read_json(path)
                details = task_manifest["splits"][split]
                self.assertEqual(details["count"], count)
                self.assertEqual(details["record_ids"], [row["id"] for row in rows])
                self.assertEqual(details["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(len(set(details["image_ids"])), count)
        stream = self.output / "taskstream_coco_hcl.json"
        loaded = DatasetLoader().load_task_stream(stream, split="test")
        self.assertEqual(len(loaded), 12)
        self.assertTrue(all(Path(example["image"]).is_absolute() for example in loaded))
        self.assertEqual(loaded[0]["metadata"]["prediction_bbox_coordinate_system"], "qwen_0_1000")
        self.assertEqual(loaded[0]["metadata"]["answer_bbox_coordinate_system"], "pixel_xyxy")
        previous = {path.relative_to(self.output): path.read_bytes() for path in self.output.rglob("*.json")}
        self.run_builder()
        self.assertEqual(previous, {
            path.relative_to(self.output): path.read_bytes() for path in self.output.rglob("*.json")
        })

    def test_insufficient_unique_images_raise_and_zero_count_is_empty(self) -> None:
        image = self.root / "image.jpg"
        image.touch()
        rows = [{"image": str(image), "metadata": {"image_id": 1}}] * 3
        self.assertEqual(builder.sample_rows(rows, 0, random.Random(630), avoid_image_ids=set()), [])
        with self.assertRaisesRegex(ValueError, "unique-image rows"):
            builder.sample_rows(rows, 2, random.Random(630), avoid_image_ids=set())

    def test_manifest_exposes_legacy_sampler_overlap(self) -> None:
        task_dir = self.output / "caption"
        paths = {}
        for split in ("train", "validation", "test"):
            path = task_dir / f"{split}.json"
            builder.write_json(path, [{"id": split, "metadata": {"image_id": 7}}])
            paths[split] = str(path.relative_to(self.output))
        with redirect_stdout(io.StringIO()) as output:
            builder.write_split_manifest(self.output, [{"task_name": "caption", "splits": paths}], 630)
        self.assertIn("WARNING", output.getvalue())
        manifest = builder.read_json(self.output / "split_manifest.json")
        self.assertEqual(manifest["tasks"]["caption"]["image_id_intersections"]["validation/test"], [7])


if __name__ == "__main__":
    unittest.main()
