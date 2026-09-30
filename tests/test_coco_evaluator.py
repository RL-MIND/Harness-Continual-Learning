from __future__ import annotations

import importlib
import importlib.util
import unittest
from unittest.mock import patch


class CocoEvaluatorRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        package = "hcl"
        cls.evaluator_module = importlib.import_module(f"{package}.evaluator")
        cls.pipeline_module = importlib.import_module(f"{package}.pipeline")

    @staticmethod
    def _example(task_id: str, task_type: str, answer: object) -> dict[str, object]:
        return {
            "task_id": task_id,
            "task_name": task_type,
            "task_type": task_type,
            "answer": answer,
            "visible_context": {
                "width": 100,
                "height": 100,
                "bbox_coordinate_system": "qwen_0_1000",
            },
            "metadata": {
                "hcl_task": task_type,
                "answer_bbox_coordinate_system": "pixel_xyxy",
                "prediction_bbox_coordinate_system": "qwen_0_1000",
            },
        }

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools is optional")
    def test_detection_uses_micro_f1_and_also_reports_coco_ap(self) -> None:
        example = self._example(
            "det-1",
            "coco_detection",
            [{"label": "person", "bbox": [10, 10, 50, 50]}],
        )
        prediction = {
            "task_id": "det-1",
            "answer": [
                {"label": "person", "bbox": [600, 600, 900, 900], "score": 0.9},
                {"label": "person", "bbox": [100, 100, 500, 500], "score": 0.8},
            ],
        }
        metrics = self.evaluator_module.ExactMatchEvaluator().evaluate(
            [example], [prediction]
        )["metrics"]

        self.assertNotIn("accuracy", metrics)
        self.assertEqual(metrics["detection_metric_source"], "pycocotools.COCOeval")
        self.assertAlmostEqual(metrics["detection_coco_ap"], 0.5)
        self.assertAlmostEqual(metrics["detection_coco_ap50"], 0.5)
        self.assertAlmostEqual(metrics["detection_micro_precision_iou50"], 0.5)
        self.assertAlmostEqual(metrics["detection_micro_recall_iou50"], 1.0)
        self.assertAlmostEqual(metrics["detection_micro_f1_iou50"], 2.0 / 3.0)
        self.assertEqual(metrics["primary_metric_name"], "detection_micro_f1_iou50")
        self.assertAlmostEqual(metrics["primary_score"], 2.0 / 3.0)

    def test_grounding_converts_qwen_coordinates_and_reports_mean_iou(self) -> None:
        example = self._example(
            "ref-1", "refcoco_grounding", {"bbox": [10, 20, 50, 80]}
        )
        prediction = {"task_id": "ref-1", "answer": {"bbox": [100, 200, 500, 800]}}
        metrics = self.evaluator_module.ExactMatchEvaluator().evaluate(
            [example], [prediction]
        )["metrics"]

        self.assertNotIn("accuracy", metrics)
        self.assertEqual(metrics["grounding_accuracy_iou50"], 1.0)
        self.assertAlmostEqual(metrics["grounding_mean_iou"], 1.0)
        self.assertEqual(metrics["primary_metric_name"], "grounding_accuracy_iou50")

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools is optional")
    def test_gate_top_level_is_current_task_and_anchors_stay_per_task(self) -> None:
        current = self._example(
            "ref-1", "refcoco_grounding", {"bbox": [10, 20, 50, 80]}
        )
        anchor = self._example(
            "det-1",
            "coco_detection",
            [{"label": "person", "bbox": [10, 10, 50, 50]}],
        )
        predictions = [
            {"task_id": "ref-1", "answer": {"bbox": [100, 200, 500, 800]}},
            {
                "task_id": "det-1",
                "answer": [
                    {"label": "person", "bbox": [100, 100, 500, 500]},
                    {"label": "dog", "bbox": [600, 600, 900, 900]},
                ],
            },
        ]
        pipeline = self.pipeline_module.HCLPipeline.__new__(self.pipeline_module.HCLPipeline)
        pipeline.evaluator = self.evaluator_module.ExactMatchEvaluator()
        result = pipeline._evaluate_gate(
            [current, anchor],
            predictions,
            current_val_examples=[current],
            historical_anchor_examples=[anchor],
            split="val",
        )

        self.assertEqual(result["metric_scope"], "current_task")
        self.assertEqual(result["metrics"]["primary_score"], 1.0)
        self.assertEqual(
            result["combined_gate_metrics"]["primary_metric_name"],
            "mixed_tasks_not_comparable",
        )
        self.assertIn("coco_detection", result["historical_anchor_task_metrics"])
        self.assertEqual(
            result["historical_anchor_metrics"]["primary_metric_name"],
            "historical_tasks_reported_separately",
        )
        self.assertIsNone(result["historical_anchor_metrics"]["primary_score"])
        self.assertIsNone(result["combined_gate_metrics"]["mean_example_score"])

    def test_caption_composite_is_named_and_not_exposed_as_accuracy(self) -> None:
        example = self._example("cap-1", "coco_caption", "a red bus")
        prediction = {"task_id": "cap-1", "answer": "a bus"}
        fake_caption_eval = (
            {"cap-1": 0.42},
            {
                "caption_metric_source": "test",
                "caption_count": 1,
                "caption_primary_score": 0.42,
                "caption_CIDEr": 0.7,
            },
        )
        with patch.object(
            self.evaluator_module,
            "_official_caption_eval_scores",
            return_value=fake_caption_eval,
        ):
            metrics = self.evaluator_module.ExactMatchEvaluator().evaluate(
                [example], [prediction]
            )["metrics"]

        self.assertNotIn("accuracy", metrics)
        self.assertEqual(metrics["primary_metric_name"], "caption_composite_score")
        self.assertEqual(metrics["primary_score"], 0.42)
        self.assertEqual(metrics["caption_CIDEr"], 0.7)

    def test_checkpoint_signatures_include_evaluation_schema(self) -> None:
        version = self.evaluator_module.EVALUATION_SCHEMA_VERSION
        train_signature = self.pipeline_module._train_checkpoint_signature(
            split="train",
            examples=[],
            val_examples=[],
            batch_count=0,
            batchsize=None,
        )
        test_signature = self.pipeline_module._test_checkpoint_signature(
            split="test", examples=[]
        )
        continual_signature = self.pipeline_module._continual_checkpoint_signature(
            task_stream_path="stream.json",
            task_order=[],
            train_split="train",
            val_split="validation",
            test_split="test",
            train_limit_per_task=None,
            validation_limit_per_task=None,
            test_limit_per_task=None,
            batchsize=None,
        )
        self.assertEqual(train_signature["evaluation_schema_version"], version)
        self.assertEqual(test_signature["evaluation_schema_version"], version)
        self.assertEqual(continual_signature["evaluation_schema_version"], version)

    def test_paper_anchor_thresholds_and_schema_validity(self) -> None:
        caption = self._example("cap", "coco_caption", "a red bus")
        vqa = self._example("vqa", "vqav2", "red")
        vqa["metadata"]["answers"] = ["red", "red", "blue"]

        self.assertEqual(self.evaluator_module._correct_threshold(caption), 0.5)
        self.assertEqual(self.evaluator_module._correct_threshold(vqa), 1.0)

        malformed_grounding = self._example(
            "ground", "refcoco_grounding", {"bbox": [10, 20, 50, 80]}
        )
        result = self.evaluator_module.ExactMatchEvaluator().evaluate(
            [malformed_grounding],
            [{"task_id": "ground", "answer": "not a box"}],
        )["metrics"]
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["format_compliant_count"], 0)


if __name__ == "__main__":
    unittest.main()
