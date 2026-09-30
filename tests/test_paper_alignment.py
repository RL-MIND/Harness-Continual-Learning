from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hcl.optimizer import Optimizer


ROOT = Path(__file__).resolve().parents[1]


class PaperAlignmentTest(unittest.TestCase):
    def test_main_stream_order_and_sample_limits(self) -> None:
        text_stream = json.loads(
            (ROOT / "configs/taskstream_textual_main_250_50_500.json").read_text()
        )
        multimodal_stream = json.loads(
            (ROOT / "configs/taskstream_multimodal_main_250_50_500.json").read_text()
        )
        self.assertEqual(
            text_stream["paper_order"],
            ["musique", "proofwriter", "gsm8k", "hotpotqa"],
        )
        self.assertEqual(
            multimodal_stream["paper_order"],
            ["coco_detection", "coco_caption", "refcoco_grounding", "vqav2"],
        )
        self.assertTrue(all("anchor" in task["splits"] for task in text_stream["tasks"]))
        self.assertTrue(
            all("anchor" in task["splits"] for task in multimodal_stream["tasks"])
        )
        for filename in (
            "deepseek_flash_reasoning_hcl_stability_250_50_500.json",
            "deepseek_flash_reasoning_hcl_plasticity_250_50_500.json",
            "qwen36_27b_coco_gpu1_hcl_stability_250_50_500.json",
            "qwen36_27b_coco_gpu1_hcl_plasticity_250_50_500.json",
        ):
            config = json.loads((ROOT / "configs" / filename).read_text())
            train = config["task_flow"][0]
            self.assertEqual(train["train_limit_per_task"], 250)
            self.assertEqual(train["validation_limit_per_task"], 50)
            self.assertEqual(train["test_limit_per_task"], 500)
            self.assertEqual(config["optimizer"]["min_format_compliance_rate"], 1.0)
            self.assertGreater(config["optimizer"]["min_primary_score_delta"], 0.0)

    def test_main_profiles_differ_only_in_historical_loss_budget(self) -> None:
        pairs = (
            (
                "deepseek_flash_reasoning_hcl_stability_250_50_500.json",
                "deepseek_flash_reasoning_hcl_plasticity_250_50_500.json",
            ),
            (
                "qwen36_27b_coco_gpu1_hcl_stability_250_50_500.json",
                "qwen36_27b_coco_gpu1_hcl_plasticity_250_50_500.json",
            ),
        )
        for stability_name, plasticity_name in pairs:
            stability = json.loads((ROOT / "configs" / stability_name).read_text())[
                "optimizer"
            ]
            plasticity = json.loads((ROOT / "configs" / plasticity_name).read_text())[
                "optimizer"
            ]
            self.assertEqual(stability.pop("historical_loss_budget"), 0)
            self.assertIsNone(plasticity.pop("historical_loss_budget"))
            self.assertEqual(stability, plasticity)

    def test_explicit_historical_budget_gates_anchor_loss(self) -> None:
        candidates = [{"candidate_id": "candidate", "validation_errors": []}]
        current = {
            "current_task_metrics": {
                "primary_score": 0.5,
                "correct": 5,
                "format_compliance_rate": 1.0,
            }
        }
        results = {
            "candidate": {
                "metrics": {},
                "current_task_metrics": {
                    "primary_score": 0.6,
                    "correct": 6,
                    "format_compliance_rate": 1.0,
                },
                "historical_anchor_metrics": {
                    "forget_count": 1,
                    "recovered_count": 0,
                    "total": 10,
                    "forget_rate": 0.1,
                },
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            stability = Optimizer(
                record_dir=Path(directory) / "stability",
                selection_objective="plasticity",
                historical_loss_budget=0,
                min_primary_score_delta=1e-9,
                min_correct_gain=0,
                min_format_compliance_rate=1.0,
            )
            plasticity = Optimizer(
                record_dir=Path(directory) / "plasticity",
                selection_objective="plasticity",
                historical_loss_budget=None,
                min_primary_score_delta=1e-9,
                min_correct_gain=0,
                min_format_compliance_rate=1.0,
            )
            self.assertIsNone(stability.select_best_candidate(candidates, results, current))
            self.assertIsNotNone(
                plasticity.select_best_candidate(candidates, results, current)
            )

    def test_budget_sweep_matches_paper_protocol(self) -> None:
        expected_budgets = {"b0": 0, "b1": 1, "b3": 3, "binf": None}
        common_optimizer = None
        for label, budget in expected_budgets.items():
            path = ROOT / "configs" / (
                f"deepseek_v4_flash_textual_budget_{label}_300_80_80_600.json"
            )
            config = json.loads(path.read_text())
            self.assertEqual(config["model"]["paper_model_name"], "DeepSeek-V4-Flash")
            self.assertEqual(config["model"]["temperature"], 0.0)
            self.assertEqual(config["model"]["thinking"], {"type": "disabled"})
            optimizer = dict(config["optimizer"])
            self.assertEqual(optimizer.pop("historical_loss_budget"), budget)
            if common_optimizer is None:
                common_optimizer = optimizer
            else:
                self.assertEqual(optimizer, common_optimizer)
            self.assertEqual(config["memory"]["anchor_capacity_per_task"], 80)
            train = config["task_flow"][0]
            self.assertEqual(300 // train["batchsize"], 10)
            self.assertEqual(train["validation_limit_per_task"], 80)
            self.assertEqual(train["test_limit_per_task"], 600)

        stream = json.loads(
            (ROOT / "configs/taskstream_textual_budget_300_80_80_600.json").read_text()
        )
        self.assertEqual(
            stream["paper_order"],
            ["musique", "proofwriter", "gsm8k", "hotpotqa"],
        )
        self.assertTrue(all("anchor" in task["splits"] for task in stream["tasks"]))

    def test_current_score_tie_prefers_lower_historical_loss(self) -> None:
        candidates = [
            {"candidate_id": "higher_loss", "validation_errors": []},
            {"candidate_id": "lower_loss", "validation_errors": []},
        ]
        current = {
            "current_task_metrics": {
                "primary_score": 0.5,
                "correct": 5,
                "format_compliance_rate": 1.0,
            }
        }
        results = {}
        for candidate_id, loss in (("higher_loss", 2), ("lower_loss", 1)):
            results[candidate_id] = {
                "metrics": {},
                "current_task_metrics": {
                    "primary_score": 0.6,
                    "correct": 6,
                    "format_compliance_rate": 1.0,
                },
                "historical_anchor_metrics": {
                    "forget_count": loss,
                    "recovered_count": 0,
                    "total": 80,
                    "forget_rate": loss / 80,
                },
            }
        with tempfile.TemporaryDirectory() as directory:
            optimizer = Optimizer(
                record_dir=directory,
                selection_objective="plasticity",
                historical_loss_budget=3,
                min_primary_score_delta=1e-9,
                min_correct_gain=0,
                min_format_compliance_rate=1.0,
            )
            selected = optimizer.select_best_candidate(candidates, results, current)
            self.assertEqual(selected["candidate_id"], "lower_loss")


if __name__ == "__main__":
    unittest.main()
