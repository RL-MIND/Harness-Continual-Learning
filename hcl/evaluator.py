from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from .json_utils import atomic_write_json


DEFAULT_EVALUATOR_DIR = Path(__file__).resolve().parent / "evaluator"
EVALUATION_SCHEMA_VERSION = "task_native_metrics_v4"


class ExactMatchEvaluator:
    def evaluate(
        self,
        examples: list[dict[str, object]],
        predictions: list[dict[str, object]],
        *,
        split: str = "",
        reference_predictions: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        by_id = {prediction.get("task_id"): prediction for prediction in predictions}
        reference_by_id = {
            prediction.get("task_id"): prediction
            for prediction in (reference_predictions or [])
        }
        caption_scores, caption_metrics = _official_caption_eval_scores(examples, by_id)
        caption_reference_scores, caption_reference_metrics = (
            _official_caption_eval_scores(examples, reference_by_id)
            if reference_predictions is not None
            else ({}, {})
        )
        correct = 0
        total = 0
        reference_correct = 0
        reference_score_sum = 0.0
        forgot = 0
        recovered = 0
        behavior_comparable = 0
        behavior_changed = 0
        format_compliant = 0
        output_chars = 0
        max_output_chars = 0
        score_sum = 0.0
        for example in examples:
            if example.get("answer") is None:
                continue
            prediction = by_id.get(example.get("task_id"))
            if prediction is None:
                continue
            total += 1
            predicted_answer = prediction.get("answer")
            task_id = str(example.get("task_id"))
            score = caption_scores.get(task_id, answer_score(predicted_answer, example))
            format_ok = answer_format_compliant(
                predicted_answer, example.get("answer"), example=example
            )
            # The paper's anchor predicate requires both task correctness and
            # a valid task-specific output schema.
            is_correct = score >= _correct_threshold(example) and format_ok
            score_sum += score
            correct += int(is_correct)
            format_compliant += int(format_ok)
            answer_chars = len(str(predicted_answer or ""))
            output_chars += answer_chars
            max_output_chars = max(max_output_chars, answer_chars)
            reference = reference_by_id.get(example.get("task_id"))
            if reference is None:
                continue
            behavior_comparable += 1
            behavior_changed += int(
                not _canonical_match(predicted_answer, reference.get("answer"))
            )
            reference_score = caption_reference_scores.get(
                task_id,
                answer_score(reference.get("answer"), example),
            )
            reference_score_sum += reference_score
            reference_format_ok = answer_format_compliant(
                reference.get("answer"), example.get("answer"), example=example
            )
            reference_is_correct = (
                reference_score >= _correct_threshold(example) and reference_format_ok
            )
            reference_correct += int(reference_is_correct)
            forgot += int(reference_is_correct and not is_correct)
            recovered += int((not reference_is_correct) and is_correct)
        mean_example_score = score_sum / total if total else 0.0
        forget_rate = forgot / reference_correct if reference_correct else 0.0
        reference_mean_example_score = (
            reference_score_sum / total
            if total and reference_predictions is not None
            else None
        )
        format_compliance_rate = format_compliant / total if total else 0.0
        metrics: dict[str, object] = {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "mean_example_score": mean_example_score,
            "correct": correct,
            "total": total,
            "forget_count": forgot,
            "forget_rate": forget_rate,
            "reference_correct": reference_correct,
            "recovered_count": recovered,
            "behavior_comparable_count": behavior_comparable,
            "behavior_change_count": behavior_changed,
            "reference_mean_example_score": reference_mean_example_score,
            "mean_example_score_delta": (
                mean_example_score - reference_mean_example_score
                if reference_mean_example_score is not None
                else None
            ),
            "has_reference": reference_predictions is not None,
            "format_compliant_count": format_compliant,
            "format_compliance_rate": format_compliance_rate,
            "average_output_chars": output_chars / total if total else 0.0,
            "max_output_chars": max_output_chars,
        }
        metrics.update(caption_metrics)
        task_metrics = _task_specific_metrics(
            examples, by_id, caption_metrics, mean_example_score=mean_example_score
        )
        metrics.update(task_metrics)
        if metrics.get("primary_metric_name") == "mixed_tasks_not_comparable":
            metrics["mean_example_score"] = None
            metrics["reference_mean_example_score"] = None
            metrics["mean_example_score_delta"] = None
        if reference_predictions is not None:
            reference_task_metrics = _task_specific_metrics(
                examples,
                reference_by_id,
                caption_reference_metrics,
                mean_example_score=reference_mean_example_score or 0.0,
            )
            metrics["reference_primary_metric_name"] = reference_task_metrics.get(
                "primary_metric_name"
            )
            metrics["reference_primary_score"] = reference_task_metrics.get("primary_score")
            primary_score = metrics.get("primary_score")
            reference_primary_score = metrics.get("reference_primary_score")
            metrics["primary_score_delta"] = (
                float(primary_score) - float(reference_primary_score)
                if primary_score is not None and reference_primary_score is not None
                else None
            )
        return {
            "split": split,
            "metrics": metrics,
            "predictions": predictions,
        }


class EvaluationRecorder:
    def __init__(
        self,
        record_dir: str | Path = DEFAULT_EVALUATOR_DIR,
        *,
        write_pretty_history: bool = True,
    ) -> None:
        self.record_dir = Path(record_dir)
        self.write_pretty_history = bool(write_pretty_history)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.record_dir / "metrics_history.jsonl"
        self.update_path = self.record_dir / "harness_update_history.jsonl"
        self.accepted_path = self.record_dir / "accepted_component_artifact_history.jsonl"
        self.rejected_path = self.record_dir / "rejected_component_artifact_history.jsonl"
        self.memory_update_path = self.record_dir / "memory_update_history.jsonl"
        self.forgetting_probe_path = self.record_dir / "forgetting_probe_history.json"
        self.task_test_path = self.record_dir / "task_test_history.json"
        self.gate_validation_path = self.record_dir / "gate_validation_history.json"
        self.accepted_artifact_dir = self.record_dir / "accepted_component_artifacts"
        self.accepted_artifact_dir.mkdir(parents=True, exist_ok=True)

    def record_metrics(
        self,
        *,
        phase: str,
        split: str,
        result: dict[str, object],
        harness_version: int,
        candidate_id: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        metrics = dict(result.get("metrics", {}))
        if phase == "candidate_val":
            metrics = compact_metrics(metrics)
        payload: dict[str, object] = {
            "phase": phase,
            "split": split,
            "harness_version": harness_version,
            "candidate_id": candidate_id,
            "artifact_id": artifact_id,
            "metrics": metrics,
        }
        for key in ("current_task_metrics", "historical_anchor_metrics"):
            scoped_metrics = result.get(key)
            if isinstance(scoped_metrics, dict):
                payload[key] = compact_metrics(scoped_metrics) if phase == "candidate_val" else scoped_metrics
        per_task_anchor_metrics = result.get("historical_anchor_task_metrics")
        if isinstance(per_task_anchor_metrics, dict):
            payload["historical_anchor_task_metrics"] = {
                str(task_name): compact_metrics(task_metrics)
                if phase == "candidate_val" and isinstance(task_metrics, dict)
                else task_metrics
                for task_name, task_metrics in per_task_anchor_metrics.items()
            }
        if result.get("metric_scope"):
            payload["metric_scope"] = result["metric_scope"]
        if phase in {"test", "task_end_test"}:
            self.record_task_test(
                {
                    "record_type": "task_test",
                    **payload,
                }
            )

    def record_forgetting_probe(self, payload: dict[str, Any]) -> None:
        self._append_json_array(
            self.forgetting_probe_path,
            compact_metric_payload({"record_type": "forgetting_probe", **payload}),
        )

    def record_task_test(self, payload: dict[str, Any]) -> None:
        self._append_json_array(
            self.task_test_path,
            compact_metric_payload({"record_type": "task_test", **payload}),
        )

    def record_gate_validation(self, payload: dict[str, Any]) -> None:
        self._append_json_array(
            self.gate_validation_path,
            compact_metric_payload({"record_type": "gate_validation", **payload}),
        )

    def record_update(self, payload: dict[str, Any]) -> None:
        self._append(self.update_path, compact_metric_payload(payload))

    def record_memory_update(self, payload: dict[str, Any]) -> None:
        self._append(self.memory_update_path, compact_metric_payload(payload))

    def record_continual_matrix(self, payload: dict[str, object]) -> str:
        path = self.record_dir / "continual_matrix.json"
        atomic_write_json(path, payload)
        return str(path)

    def persist_accepted_artifact(
        self,
        *,
        harness_version: int,
        candidate: dict[str, object],
    ) -> str:
        component = str(candidate.get("component") or "component")
        path = self.accepted_artifact_dir / f"harness_v{harness_version}_{component}.json"
        artifact_content = candidate.get("artifact_content")
        if not isinstance(artifact_content, dict):
            artifact_content = {}
        atomic_write_json(path, artifact_content)
        return str(path)

    def record_accepted_component_artifact(self, payload: dict[str, Any]) -> None:
        self._append(self.accepted_path, compact_metric_payload(payload))

    def record_rejected_component_artifact(self, payload: dict[str, Any]) -> None:
        self._append(self.rejected_path, compact_metric_payload(payload))

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if not self.write_pretty_history:
            return
        pretty_path = path.with_suffix(".json")
        records: list[Any] = []
        if pretty_path.exists():
            try:
                loaded = json.loads(pretty_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    records = loaded
            except json.JSONDecodeError:
                records = []
        records.append(payload)
        atomic_write_json(pretty_path, records)

    @staticmethod
    def _append_json_array(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records: list[Any] = []
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    records = loaded
            except json.JSONDecodeError:
                records = []
        records.append(payload)
        atomic_write_json(path, records)




def compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    compacted = {
        "evaluation_schema_version": metrics.get("evaluation_schema_version"),
        "primary_metric_name": metrics.get("primary_metric_name"),
        "primary_score": metrics.get("primary_score"),
        "primary_score_delta": metrics.get("primary_score_delta"),
        "mean_example_score": metrics.get("mean_example_score", 0.0),
        "correct": metrics.get("correct", 0),
        "total": metrics.get("total", 0),
        "format_compliance_rate": metrics.get("format_compliance_rate", 0.0),
        "average_output_chars": metrics.get("average_output_chars", 0.0),
        "max_output_chars": metrics.get("max_output_chars", 0),
        "forget_count": metrics.get("forget_count", 0),
        "forget_rate": metrics.get("forget_rate", 0.0),
        "recovered_count": metrics.get("recovered_count", 0),
        "behavior_comparable_count": metrics.get("behavior_comparable_count", 0),
        "behavior_change_count": metrics.get("behavior_change_count", 0),
    }
    for key, value in metrics.items():
        if str(key).startswith(("caption_", "detection_", "grounding_", "vqa_", "primary_")):
            compacted[str(key)] = value
    return compacted


def compact_metric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in payload.items():
        if key.endswith("metrics") and isinstance(value, dict):
            compacted[key] = compact_metrics(value)
        else:
            compacted[key] = value
    return compacted
def normalize(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def answer_matches(prediction: object, target: object, *, example: dict[str, object] | None = None) -> bool:
    """Exact match with only the dataset's explicit answer-format conventions."""
    if isinstance(example, dict):
        return answer_score(prediction, example) >= _correct_threshold(example)
    targets = [target]
    return any(_canonical_match(prediction, candidate) for candidate in targets if candidate is not None)


def _canonical_match(prediction: object, target: object) -> bool:
    pred = _strip_answer_wrapper(str(prediction or ""))
    gold = str(target or "").strip()
    if "####" in gold:
        gold = gold.rsplit("####", 1)[-1].strip()
    if re.fullmatch(r"[A-Ea-e]", gold):
        choice = _single_choice(pred)
        return choice is not None and choice.lower() == gold.lower()
    pred_number = _single_number(pred)
    gold_number = _single_number(gold)
    if pred_number is not None and gold_number is not None:
        return pred_number == gold_number
    return normalize(pred) == normalize(gold)


def answer_score(prediction: object, example: dict[str, object]) -> float:
    task_kind = _task_kind(example)
    target = example.get("answer")
    metadata = example.get("metadata", {})
    if task_kind == "coco_detection":
        return _detection_score(prediction, target, example)
    if task_kind == "refcoco_grounding":
        pred_bbox = _extract_bbox(prediction)
        gold_bbox = _extract_bbox(target)
        if pred_bbox is None or gold_bbox is None:
            return 0.0
        pred_bbox = _prediction_bbox_to_image_pixels(pred_bbox, example)
        return 1.0 if _bbox_iou(pred_bbox, gold_bbox) >= 0.5 else 0.0
    if task_kind == "coco_caption":
        references = metadata.get("answers") if isinstance(metadata, dict) else None
        references = references if isinstance(references, list) else [target]
        return max((_caption_token_f1(prediction, ref) for ref in references), default=0.0)
    if task_kind in {"vqa", "vqav2"}:
        answers = metadata.get("answers") if isinstance(metadata, dict) else None
        answers = answers if isinstance(answers, list) and answers else [target]
        pred = normalize(_strip_answer_wrapper(str(prediction or "")))
        matches = sum(1 for answer in answers if normalize(answer) == pred)
        return min(matches / 3.0, 1.0)
    targets = [target]
    if isinstance(metadata, dict) and isinstance(metadata.get("answers"), list):
        targets.extend(metadata["answers"])
    return 1.0 if any(
        _canonical_match(prediction, candidate)
        for candidate in targets
        if candidate is not None
    ) else 0.0


def answer_format_compliant(
    prediction: object,
    target: object,
    *,
    example: dict[str, object] | None = None,
) -> bool:
    """Check the final-only output contract independently of answer correctness."""
    raw = str(prediction or "").strip()
    if not raw:
        return False
    if isinstance(example, dict):
        task_kind = _task_kind(example)
        if task_kind == "coco_detection":
            return bool(_extract_object_list(prediction))
        if task_kind == "refcoco_grounding":
            return _extract_bbox(prediction) is not None
        if task_kind == "coco_caption":
            return "\n" not in raw and len(raw) <= 256 and 3 <= len(_caption_tokens(raw)) <= 30
        if task_kind in {"vqa", "vqav2"}:
            return "\n" not in raw and len(raw) <= 256 and len(raw.split()) <= 5
    if "\n" in raw or len(raw) > 256:
        return False
    pred = _strip_answer_wrapper(raw)
    gold = str(target or "").strip()
    if "####" in gold:
        gold = gold.rsplit("####", 1)[-1].strip()
    if re.fullmatch(r"[A-Ea-e]", gold):
        return _single_choice(pred) is not None
    if _single_number(gold) is not None:
        return _single_number(pred) is not None
    return bool(pred) and len(pred) <= 128


def _strip_answer_wrapper(value: str) -> str:
    text = value.strip()
    patterns = (
        r"^####\s*(.+?)\s*$",
        r"^(?:final\s+answer|answer)\s*:\s*(.+?)\s*$",
        r"^the\s+answer\s+is\s+(.+?)\s*$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def _single_choice(value: str) -> str | None:
    match = re.fullmatch(r"\(?\s*([A-Ea-e])\s*\)?[\.!]?", value.strip())
    return match.group(1) if match else None


def _single_number(value: str) -> str | None:
    compact = value.replace(",", "").strip().rstrip(".")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", compact):
        return None
    try:
        return str(float(compact))
    except ValueError:
        return None


def _task_kind(example: dict[str, object]) -> str:
    metadata = example.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("hcl_task"):
        return str(metadata["hcl_task"]).lower()
    return str(example.get("task_type", "")).lower()


def _correct_threshold(example: dict[str, object]) -> float:
    task_kind = _task_kind(example)
    if task_kind == "coco_caption":
        # Appendix G defines caption-anchor success as sentence-level
        # CIDEr >= 0.5 on the normalized [0, 1] scale.
        return 0.50
    if task_kind == "coco_detection":
        return 0.5
    if task_kind in {"vqa", "vqav2"}:
        # Appendix G requires the standard VQA consensus score to be 1.0.
        return 1.0
    return 1.0


def _official_caption_eval_scores(
    examples: list[dict[str, object]],
    predictions_by_id: dict[object, dict[str, object]],
) -> tuple[dict[str, float], dict[str, object]]:
    caption_examples = [
        example
        for example in examples
        if _task_kind(example) == "coco_caption"
        and example.get("answer") is not None
        and predictions_by_id.get(example.get("task_id")) is not None
    ]
    if not caption_examples:
        return {}, {}
    gts: dict[str, list[dict[str, str]]] = {}
    res: dict[str, list[dict[str, str]]] = {}
    for example in caption_examples:
        task_id = str(example.get("task_id"))
        metadata = example.get("metadata", {})
        references = metadata.get("answers") if isinstance(metadata, dict) else None
        references = references if isinstance(references, list) and references else [example.get("answer")]
        gts[task_id] = [
            {"caption": str(reference or "")}
            for reference in references
            if str(reference or "").strip()
        ]
        prediction = predictions_by_id.get(example.get("task_id"), {})
        res[task_id] = [{"caption": _strip_answer_wrapper(str(prediction.get("answer") or ""))}]
    if not gts:
        return {}, {}
    try:
        tokenized_gts = _ptb_tokenize(gts)
        tokenized_res = _ptb_tokenize(res)
        per_metric_scores, aggregate = _compute_coco_caption_metrics(tokenized_gts, tokenized_res)
    except Exception as exc:
        fallback_scores = {
            task_id: max(
                (_caption_token_f1(res[task_id][0]["caption"], ref["caption"]) for ref in refs),
                default=0.0,
            )
            for task_id, refs in gts.items()
        }
        # Token F1 is useful as a diagnostic only; it is not the paper's
        # caption-anchor predicate and must not silently admit an anchor.
        anchor_scores = {task_id: 0.0 for task_id in gts}
        return anchor_scores, {
            "caption_eval_error": f"{type(exc).__name__}:{exc}",
            "caption_metric_source": "fallback_token_f1",
            "caption_anchor_criterion_available": False,
            "caption_count": len(gts),
            "caption_primary_score": (
                sum(fallback_scores.values()) / len(fallback_scores)
                if fallback_scores
                else 0.0
            ),
        }
    primary_scores: dict[str, float] = {}
    for task_id in gts:
        components = []
        for metric_name in ("Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"):
            metric_scores = per_metric_scores.get(metric_name)
            if metric_scores is None or task_id not in metric_scores:
                continue
            components.append(min(max(float(metric_scores[task_id]), 0.0), 1.0))
        primary_scores[task_id] = sum(components) / len(components) if components else 0.0
    cider_scores = per_metric_scores.get("CIDEr", {})
    anchor_scores = {
        task_id: min(max(float(cider_scores.get(task_id, 0.0)), 0.0), 1.0)
        for task_id in gts
    }
    metrics: dict[str, object] = {
        "caption_metric_source": "pycocoevalcap",
        "caption_anchor_criterion_available": True,
        "caption_anchor_metric_definition": "sentence_level_cider_at_least_0.5",
        "caption_count": len(gts),
        "caption_primary_score": (
            sum(primary_scores.values()) / len(primary_scores)
            if primary_scores
            else 0.0
        ),
    }
    for name, value in aggregate.items():
        metrics[f"caption_{name}"] = value
    return anchor_scores, metrics


def _compute_coco_caption_metrics(
    gts: dict[str, list[str]],
    res: dict[str, list[str]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    scorers: list[tuple[object, list[str] | str]] = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    if os.environ.get("HCL_CAPTION_SPICE") == "1":
        from pycocoevalcap.spice.spice import Spice

        scorers.append((Spice(), "SPICE"))
    per_metric: dict[str, dict[str, float]] = {}
    aggregate: dict[str, float] = {}
    task_ids = list(gts.keys())
    for scorer, method in scorers:
        try:
            with redirect_stdout(StringIO()):
                score, scores = scorer.compute_score(gts, res)
        finally:
            _close_caption_scorer(scorer)
        if isinstance(method, list):
            for metric_name, metric_score, metric_scores in zip(method, score, scores):
                aggregate[metric_name] = float(metric_score)
                per_metric[metric_name] = {
                    task_id: float(metric_scores[index])
                    for index, task_id in enumerate(task_ids)
                }
        else:
            aggregate[method] = float(score)
            per_metric[method] = {
                task_id: float(scores[index])
                for index, task_id in enumerate(task_ids)
            }
    return per_metric, aggregate


def _close_caption_scorer(scorer: object) -> None:
    if hasattr(scorer, "close"):
        scorer.close()
        return
    meteor_p = getattr(scorer, "meteor_p", None)
    if meteor_p is None:
        return
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(meteor_p, stream_name, None)
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass
    try:
        meteor_p.kill()
        meteor_p.wait()
    except OSError:
        pass


def _ptb_tokenize(
    captions_for_image: dict[str, list[dict[str, str]]],
) -> dict[str, list[str]]:
    import pycocoevalcap.tokenizer.ptbtokenizer as ptb

    image_ids = [key for key, values in captions_for_image.items() for _ in range(len(values))]
    sentences = "\n".join(
        item["caption"].replace("\n", " ")
        for values in captions_for_image.values()
        for item in values
    )
    jar_path = Path(ptb.__file__).resolve().parent / ptb.STANFORD_CORENLP_3_4_1_JAR
    with tempfile.NamedTemporaryFile(delete=False, dir="/tmp") as tmp:
        tmp.write(sentences.encode("utf-8"))
        tmp_path = tmp.name
    try:
        proc = subprocess.Popen(
            [
                "java",
                "-cp",
                str(jar_path),
                "edu.stanford.nlp.process.PTBTokenizer",
                "-preserveLines",
                "-lowerCase",
                tmp_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    tokenized: dict[str, list[str]] = {}
    for key, line in zip(image_ids, stdout.decode("utf-8", errors="replace").split("\n")):
        tokenized.setdefault(key, []).append(
            " ".join(word for word in line.rstrip().split(" ") if word not in ptb.PUNCTUATIONS)
        )
    return tokenized


def _detection_score(
    prediction: object,
    target: object,
    example: dict[str, object] | None = None,
) -> float:
    matched, predicted, gold = _detection_match_counts(prediction, target, example)
    # Per-image F1 makes both missed gold objects and false-positive
    # predictions visible to the HCL gate.  The previous matched/gold score
    # was recall-only and allowed arbitrarily many false positives for free.
    return 2.0 * matched / (predicted + gold) if predicted + gold else 0.0


def _detection_match_counts(
    prediction: object,
    target: object,
    example: dict[str, object] | None = None,
) -> tuple[int, int, int]:
    pred_objects = _extract_object_list(prediction)
    gold_objects = _extract_object_list(target)
    if not pred_objects or not gold_objects:
        return 0, len(pred_objects), len(gold_objects)
    force_prediction_scale = (
        _prediction_bbox_coordinate_system(example) == "qwen_0_1000"
        if example is not None
        else False
    )
    if example is not None:
        force_prediction_scale = force_prediction_scale or any(
            _bbox_looks_normalized_1000(_extract_bbox(pred), example)
            for pred in pred_objects
        )
    matched_pred: set[int] = set()
    matched = 0
    for gold in gold_objects:
        gold_label = normalize(gold.get("label"))
        gold_bbox = _extract_bbox(gold)
        if gold_bbox is None:
            continue
        best_index = None
        best_iou = 0.0
        for index, pred in enumerate(pred_objects):
            if index in matched_pred or normalize(pred.get("label")) != gold_label:
                continue
            pred_bbox = _extract_bbox(pred)
            if pred_bbox is None:
                continue
            if example is not None:
                pred_bbox = _prediction_bbox_to_image_pixels(
                    pred_bbox,
                    example,
                    force=force_prediction_scale,
                )
            iou = _bbox_iou(pred_bbox, gold_bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index is not None and best_iou >= 0.5:
            matched += 1
            matched_pred.add(best_index)
    return matched, len(pred_objects), len(gold_objects)


def _coco_detection_metrics(
    examples: list[dict[str, object]],
    predictions_by_id: dict[object, dict[str, object]],
) -> dict[str, object]:
    """Evaluate detections with the canonical pycocotools COCOeval implementation."""
    labels = sorted({
        normalize(obj.get("label"))
        for example in examples
        for obj in (
            _extract_object_list(example.get("answer"))
            + _extract_object_list(
                predictions_by_id.get(example.get("task_id"), {}).get("answer")
            )
        )
        if normalize(obj.get("label"))
    })
    category_ids = {label: index + 1 for index, label in enumerate(labels)}
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    detections: list[dict[str, object]] = []
    annotation_id = 1
    for image_id, example in enumerate(examples, start=1):
        images.append({"id": image_id})
        for obj in _extract_object_list(example.get("answer")):
            label = normalize(obj.get("label"))
            bbox = _extract_bbox(obj)
            if not label or bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[label],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
            annotation_id += 1
        pred_objects = _extract_object_list(
            predictions_by_id.get(example.get("task_id"), {}).get("answer")
        )
        force_prediction_scale = _prediction_bbox_coordinate_system(example) == "qwen_0_1000"
        force_prediction_scale = force_prediction_scale or any(
            _bbox_looks_normalized_1000(_extract_bbox(obj), example)
            for obj in pred_objects
        )
        for obj in pred_objects:
            label = normalize(obj.get("label"))
            bbox = _extract_bbox(obj)
            if not label or bbox is None:
                continue
            bbox = _prediction_bbox_to_image_pixels(
                bbox, example, force=force_prediction_scale
            )
            x1, y1, x2, y2 = bbox
            detections.append({
                "image_id": image_id,
                "category_id": category_ids[label],
                "bbox": [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)],
                "score": _detection_confidence(obj),
            })

    empty = {
        "detection_metric_source": "pycocotools.COCOeval",
        "detection_coco_metric_definition": "AP@[IoU=0.50:0.95|area=all|maxDets=100]",
        "detection_predictions_without_confidence": sum(
            not any(key in obj for key in ("score", "confidence", "probability"))
            for example in examples
            for obj in _extract_object_list(
                predictions_by_id.get(example.get("task_id"), {}).get("answer")
            )
        ),
        "detection_coco_ap": 0.0,
        "detection_coco_ap50": 0.0,
        "detection_coco_ap75": 0.0,
        "detection_coco_ap_small": 0.0,
        "detection_coco_ap_medium": 0.0,
        "detection_coco_ap_large": 0.0,
        "detection_coco_ar_max100": 0.0,
    }
    if not annotations or not detections or not category_ids:
        return empty

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        with redirect_stdout(StringIO()):
            ground_truth = COCO()
            ground_truth.dataset = {
                "info": {},
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": [
                    {"id": category_id, "name": label, "supercategory": "object"}
                    for label, category_id in category_ids.items()
                ],
            }
            ground_truth.createIndex()
            detected = ground_truth.loadRes(detections)
            coco_eval = COCOeval(ground_truth, detected, "bbox")
            coco_eval.params.imgIds = [int(image["id"]) for image in images]
            coco_eval.params.catIds = list(category_ids.values())
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        stats = [max(float(value), 0.0) for value in coco_eval.stats]
    except Exception as exc:
        raise RuntimeError(
            "COCO detection evaluation requires a working pycocotools.COCOeval"
        ) from exc
    return {
        **empty,
        "detection_coco_ap": stats[0],
        "detection_coco_ap50": stats[1],
        "detection_coco_ap75": stats[2],
        "detection_coco_ap_small": stats[3],
        "detection_coco_ap_medium": stats[4],
        "detection_coco_ap_large": stats[5],
        "detection_coco_ar_max100": stats[8],
    }


def _detection_confidence(obj: dict[str, object]) -> float:
    for key in ("score", "confidence", "probability"):
        value = obj.get(key)
        try:
            return min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            continue
    # Older model outputs did not require confidence. COCOeval still accepts
    # them, while the separately reported F1 remains insensitive to ranking.
    return 1.0


def _task_specific_metrics(
    examples: list[dict[str, object]],
    predictions_by_id: dict[object, dict[str, object]],
    caption_metrics: dict[str, object],
    *,
    mean_example_score: float,
) -> dict[str, object]:
    """Return a named, task-native primary metric and useful diagnostics."""
    evaluated = [
        example
        for example in examples
        if example.get("answer") is not None
        and predictions_by_id.get(example.get("task_id")) is not None
    ]
    kinds = {_task_kind(example) for example in evaluated}
    result: dict[str, object] = {}

    detection_examples = [example for example in evaluated if _task_kind(example) == "coco_detection"]
    if detection_examples:
        matched = predicted = gold = 0
        image_f1_sum = 0.0
        for example in detection_examples:
            prediction = predictions_by_id[example.get("task_id")].get("answer")
            item_matched, item_predicted, item_gold = _detection_match_counts(
                prediction, example.get("answer"), example
            )
            matched += item_matched
            predicted += item_predicted
            gold += item_gold
            image_f1_sum += (
                2.0 * item_matched / (item_predicted + item_gold)
                if item_predicted + item_gold
                else 0.0
            )
        precision = matched / predicted if predicted else 0.0
        recall = matched / gold if gold else 0.0
        micro_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.update({
            "detection_metric_definition": "label_matched_f1_at_iou_0.5",
            "detection_image_count": len(detection_examples),
            "detection_matched_count": matched,
            "detection_predicted_count": predicted,
            "detection_gold_count": gold,
            "detection_false_positive_count": max(predicted - matched, 0),
            "detection_false_negative_count": max(gold - matched, 0),
            "detection_micro_precision_iou50": precision,
            "detection_micro_recall_iou50": recall,
            "detection_micro_f1_iou50": micro_f1,
            "detection_mean_image_f1_iou50": image_f1_sum / len(detection_examples),
        })
        result.update(_coco_detection_metrics(detection_examples, predictions_by_id))

    grounding_examples = [example for example in evaluated if _task_kind(example) == "refcoco_grounding"]
    if grounding_examples:
        ious: list[float] = []
        for example in grounding_examples:
            prediction = predictions_by_id[example.get("task_id")].get("answer")
            pred_bbox = _extract_bbox(prediction)
            gold_bbox = _extract_bbox(example.get("answer"))
            if pred_bbox is None or gold_bbox is None:
                ious.append(0.0)
                continue
            ious.append(_bbox_iou(_prediction_bbox_to_image_pixels(pred_bbox, example), gold_bbox))
        result.update({
            "grounding_metric_definition": "accuracy_at_iou_0.5",
            "grounding_count": len(ious),
            "grounding_mean_iou": sum(ious) / len(ious),
            "grounding_accuracy_iou50": sum(iou >= 0.5 for iou in ious) / len(ious),
            "grounding_accuracy_iou75": sum(iou >= 0.75 for iou in ious) / len(ious),
        })

    if len(kinds) == 1:
        kind = next(iter(kinds))
        if kind == "coco_detection":
            result["primary_metric_name"] = "detection_micro_f1_iou50"
            result["primary_score"] = result["detection_micro_f1_iou50"]
        elif kind == "refcoco_grounding":
            result["primary_metric_name"] = "grounding_accuracy_iou50"
            result["primary_score"] = result["grounding_accuracy_iou50"]
        elif kind == "coco_caption":
            result["primary_metric_name"] = "caption_composite_score"
            result["primary_score"] = caption_metrics.get("caption_primary_score", 0.0)
        elif kind in {"vqa", "vqav2"}:
            result["primary_metric_name"] = "vqa_consensus_accuracy"
            # VQAv2 uses the official min(matches / 3, 1) consensus score.
            result["primary_score"] = sum(
                answer_score(
                    predictions_by_id[example.get("task_id")].get("answer"), example
                )
                for example in evaluated
            ) / len(evaluated)
            result["vqa_metric_definition"] = "min_matching_annotators_over_3"
        else:
            result["primary_metric_name"] = "exact_match_accuracy"
            result["primary_score"] = mean_example_score
    elif kinds:
        result["primary_metric_name"] = "mixed_tasks_not_comparable"
        result["primary_score"] = None
        result["mixed_task_kinds"] = sorted(kinds)
    else:
        result["primary_metric_name"] = "no_evaluable_examples"
        result["primary_score"] = None
    return result


def _extract_object_list(value: object) -> list[dict[str, object]]:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, dict):
        for key in ("objects", "detections", "predictions"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        return []
    objects = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("category") or item.get("name")
        bbox = _extract_bbox(item)
        if label and bbox is not None:
            normalized: dict[str, object] = {"label": str(label), "bbox": bbox}
            for confidence_key in ("score", "confidence", "probability"):
                if confidence_key in item:
                    normalized[confidence_key] = item[confidence_key]
                    break
            objects.append(normalized)
    return objects


def _extract_bbox(value: object) -> list[float] | None:
    parsed = _parse_jsonish(value)
    raw_bbox = (
        parsed.get("bbox") or parsed.get("box") or parsed.get("bbox_2d")
        if isinstance(parsed, dict)
        else parsed
    )
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in raw_bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _parse_jsonish(value: object) -> object:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [
            index
            for index in (text.find("{"), text.find("["))
            if index >= 0
        ]
        if not start_candidates:
            return value
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[min(start_candidates):])
            return parsed
        except json.JSONDecodeError:
            return value


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _prediction_bbox_to_image_pixels(
    bbox: list[float],
    example: dict[str, object],
    *,
    force: bool = False,
) -> list[float]:
    width, height = _image_size(example)
    if width is None or height is None:
        return bbox
    coordinate_system = _prediction_bbox_coordinate_system(example)
    if force or coordinate_system == "qwen_0_1000" or _bbox_looks_normalized_1000(bbox, example):
        return [
            bbox[0] / 1000.0 * width,
            bbox[1] / 1000.0 * height,
            bbox[2] / 1000.0 * width,
            bbox[3] / 1000.0 * height,
        ]
    return bbox


def _prediction_bbox_coordinate_system(example: dict[str, object] | None) -> str:
    if not isinstance(example, dict):
        return ""
    metadata = example.get("metadata", {})
    visible_context = example.get("visible_context", {})
    for values in (metadata, visible_context):
        if not isinstance(values, dict):
            continue
        coordinate_system = values.get("prediction_bbox_coordinate_system") or values.get("bbox_coordinate_system")
        if coordinate_system:
            return str(coordinate_system).lower()
    return ""


def _bbox_looks_normalized_1000(
    bbox: list[float] | None,
    example: dict[str, object],
) -> bool:
    if bbox is None or min(bbox) < 0.0 or max(bbox) > 1000.0:
        return False
    width, height = _image_size(example)
    if width is None or height is None:
        return False
    return max(bbox[0], bbox[2]) > width * 1.05 or max(bbox[1], bbox[3]) > height * 1.05


def _image_size(example: dict[str, object]) -> tuple[float | None, float | None]:
    for values in (example.get("visible_context", {}), example.get("metadata", {})):
        if not isinstance(values, dict):
            continue
        width = _as_float(values.get("width") or values.get("image_width"))
        height = _as_float(values.get("height") or values.get("image_height"))
        if width and height:
            return width, height
    return None, None


def _as_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _caption_token_f1(prediction: object, reference: object) -> float:
    pred_tokens = _caption_tokens(_strip_answer_wrapper(str(prediction or "")))
    ref_tokens = _caption_tokens(str(reference or ""))
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = sum(min(count, ref_counts.get(token, 0)) for token, count in pred_counts.items())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _caption_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())
