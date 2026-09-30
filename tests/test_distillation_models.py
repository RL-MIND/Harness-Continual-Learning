from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hcl.evaluator import EvaluationRecorder
from hcl.memory import MemoryConfig
from hcl.optimizer import Optimizer
from hcl.pipeline import HCLPipeline


class _Model:
    def generate(self, prompt: str, *, state: dict[str, object]) -> str:
        del prompt, state
        return "{}"


class DistillationModelRoutingTest(unittest.TestCase):
    def _pipeline(
        self,
        directory: str,
        *,
        model: _Model,
        judge_model: _Model | None = None,
        selection_model: _Model | None = None,
        memory_model: _Model | None = None,
        judge_memory: bool = False,
    ) -> HCLPipeline:
        return HCLPipeline(
            model,
            judge_model=judge_model,
            selection_model=selection_model,
            memory_model=memory_model,
            judge_memory=judge_memory,
            evaluation_recorder=EvaluationRecorder(Path(directory) / "evaluator"),
            optimizer=Optimizer(record_dir=Path(directory) / "optimizer", candidate_count=0),
            memory_config=MemoryConfig(enabled=False, record_dir=Path(directory) / "memory"),
        )

    def test_judge_defaults_to_main_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _Model()
            pipeline = self._pipeline(directory, model=model)

        self.assertIs(pipeline.judge_model, model)
        self.assertIs(pipeline.selection_model, model)
        self.assertIs(pipeline.memory_model, model)
        self.assertIs(pipeline.memory.model, model)

    def test_partial_distillation_keeps_memory_on_main_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _Model()
            judge = _Model()
            pipeline = self._pipeline(directory, model=model, judge_model=judge)

        self.assertIs(pipeline.judge_model, judge)
        self.assertIs(pipeline.memory.model, model)

    def test_full_distillation_routes_memory_to_judge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _Model()
            judge = _Model()
            pipeline = self._pipeline(
                directory,
                model=model,
                judge_model=judge,
                judge_memory=True,
            )

        self.assertIs(pipeline.judge_model, judge)
        self.assertIs(pipeline.memory.model, judge)

    def test_online_teacher_routes_selection_and_memory_but_not_final_or_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _Model()
            teacher = _Model()
            pipeline = self._pipeline(
                directory,
                model=model,
                selection_model=teacher,
                memory_model=teacher,
            )

        self.assertIs(pipeline.model, model)
        self.assertIs(pipeline.judge_model, model)
        self.assertIs(pipeline.selection_model, teacher)
        self.assertIs(pipeline.memory_model, teacher)
        self.assertIs(pipeline.memory.model, teacher)


if __name__ == "__main__":
    unittest.main()
