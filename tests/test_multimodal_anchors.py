from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hcl.memory.anchor import AnchorMemory


class MultimodalAnchorTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.memory = AnchorMemory(Path(directory.name) / "anchors.jsonl", capacity_per_task=25)

    @staticmethod
    def example(index: int, **changes: object) -> dict:
        return {
            "task_id": f"caption/train/{index}",
            "task_name": "caption",
            "question": "Describe this image.",
            "image": f"not-mounted/image-{index}.jpg",
            **changes,
        }

    def test_25_same_template_images_are_retained_without_reading_files(self) -> None:
        with patch.object(Path, "read_bytes", side_effect=AssertionError("image read")):
            self.memory.add("caption", [self.example(index) for index in range(25)])
        self.assertEqual(len(self.memory.all_examples()), 25)

    def test_content_and_path_duplicates_with_distinct_ids_are_retained(self) -> None:
        self.memory.add("caption", [self.example(index, image="same.jpg") for index in range(5)])
        self.assertEqual(len(self.memory.all_examples()), 5)

    def test_twenty_anchors_accumulate_across_ten_batches(self) -> None:
        for batch in range(10):
            added = self.memory.add_batch(
                "caption", [self.example(batch * 2 + offset) for offset in range(2)]
            )
            self.assertEqual(len(added), 2)
        self.assertEqual(len(self.memory.all_examples()), 20)
        restored = AnchorMemory(self.memory.path, capacity_per_task=25)
        self.assertEqual(len(restored.all_examples()), 20)
        self.assertEqual(restored.add_batch("caption", restored.all_examples()), [])

    def test_attachment_aliases_bypass_text_deduplication(self) -> None:
        for field in ("image_path", "file", "file_path"):
            with self.subTest(field=field):
                examples = []
                for index in range(3):
                    example = self.example(index)
                    example[field] = example.pop("image")
                    examples.append(example)
                self.memory.add("caption", examples)
                self.assertEqual(len(self.memory.all_examples()), 3)

    def test_text_only_deduplication_is_unchanged(self) -> None:
        examples = []
        for index in range(3):
            example = self.example(index)
            del example["image"]
            examples.append(example)
        self.memory.add("caption", examples)
        self.assertEqual(len(self.memory.all_examples()), 1)

    def test_capacity_and_other_task_records_are_preserved(self) -> None:
        other = {"task_id": "math/1", "task_name": "math", "question": "1 + 1?"}
        self.memory.add("math", [other])
        self.memory.add_batch(
            "caption", [self.example(index) for index in range(10)], max_total_per_task=4,
        )
        self.assertIn(other, self.memory.all_examples())
        self.assertEqual(len(self.memory.all_examples()), 5)


if __name__ == "__main__":
    unittest.main()
