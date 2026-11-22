"""Recovered COCO/RefCOCO/VQAv2 record construction and task prompts."""
from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def detection_rows(coco: dict[str, Any], image_dir: Path, categories: dict[int, str], *, split: str) -> list[dict[str, Any]]:
    images = {int(item["id"]): item for item in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[int(ann["image_id"])].append(ann)
    rows = []
    for image_id, anns in anns_by_image.items():
        image = images.get(image_id)
        if not image:
            continue
        objects = []
        for ann in anns[:20]:
            x, y, w, h = [float(value) for value in ann["bbox"]]
            objects.append({"label": categories.get(int(ann["category_id"]), "object"), "bbox": [x, y, x + w, y + h]})
        if not objects:
            continue
        rows.append(
            make_example(
                f"det-{split}-{image_id}",
                "Detect the main COCO objects in the image. Return JSON only as a list of objects with label and bbox fields. Use bbox format [x1, y1, x2, y2] in the 0-1000 image coordinate system.",
                image_dir / image["file_name"],
                objects,
                "coco_detection",
                image_id,
                split,
                visible_context={
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "output_format": "json_objects_xyxy_0_1000",
                    "bbox_coordinate_system": "qwen_0_1000",
                },
                metadata={
                    "answers": [objects],
                    "target_objects": objects,
                    "answer_bbox_coordinate_system": "pixel_xyxy",
                    "prediction_bbox_coordinate_system": "qwen_0_1000",
                },
            )
        )
    return rows


def caption_rows(coco: dict[str, Any], image_dir: Path, *, split: str) -> list[dict[str, Any]]:
    images = {int(item["id"]): item for item in coco["images"]}
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for ann in coco["annotations"]:
        captions_by_image[int(ann["image_id"])].append(str(ann["caption"]).strip())
    rows = []
    for image_id, captions in captions_by_image.items():
        image = images.get(image_id)
        if not image or not captions:
            continue
        rows.append(
            make_example(
                f"cap-{split}-{image_id}",
                "Write one short natural-language caption describing the main visible content of the image.",
                image_dir / image["file_name"],
                captions[0],
                "coco_caption",
                image_id,
                split,
                metadata={"answers": captions},
            )
        )
    return rows


def refcoco_rows(instances_path: Path, refs_path: Path, image_dir: Path) -> list[dict[str, Any]]:
    instances = load_coco(instances_path)
    images = {int(item["id"]): item for item in instances["images"]}
    anns = {int(item["id"]): item for item in instances["annotations"]}
    categories = {int(item["id"]): item["name"] for item in instances["categories"]}
    with refs_path.open("rb") as f:
        refs = pickle.load(f)
    rows = []
    for ref in refs:
        ann = anns.get(int(ref["ann_id"]))
        image = images.get(int(ref["image_id"]))
        if not ann or not image:
            continue
        x, y, w, h = [float(value) for value in ann["bbox"]]
        bbox = [x, y, x + w, y + h]
        expression = str(ref.get("sentences", [{}])[0].get("sent", "")).strip()
        if not expression:
            continue
        split = str(ref.get("split", "train"))
        rows.append(
            make_example(
                f"refcoco-{split}-{ref['ref_id']}",
                f"Locate the object described by this referring expression: {expression}. Return JSON only as {{\"bbox\": [x1, y1, x2, y2]}} using the 0-1000 image coordinate system.",
                image_dir / image["file_name"],
                {"bbox": bbox},
                "refcoco_grounding",
                int(ref["image_id"]),
                split,
                visible_context={
                    "expression": expression,
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "output_format": "json_bbox_xyxy_0_1000",
                    "bbox_coordinate_system": "qwen_0_1000",
                },
                metadata={
                    "answers": [{"bbox": bbox}],
                    "answer_bbox_coordinate_system": "pixel_xyxy",
                    "prediction_bbox_coordinate_system": "qwen_0_1000",
                    "category": categories.get(int(ann["category_id"]), "object"),
                    "ref_split": split,
                    "expression": expression,
                },
            )
        )
    return rows


def vqa_rows(questions_path: Path, annotations_path: Path, image_dir: Path, *, split: str) -> list[dict[str, Any]]:
    questions = read_json(questions_path)["questions"]
    annotations = {int(item["question_id"]): item for item in read_json(annotations_path)["annotations"]}
    rows = []
    prefix = "COCO_train2014" if split == "train" else "COCO_val2014"
    for question in questions:
        ann = annotations.get(int(question["question_id"]))
        if not ann:
            continue
        answers = [str(item["answer"]).strip() for item in ann.get("answers", []) if str(item.get("answer", "")).strip()]
        if not answers:
            continue
        answer = Counter(answers).most_common(1)[0][0]
        image_id = int(question["image_id"])
        image_name = f"{prefix}_{image_id:012d}.jpg"
        rows.append(
            make_example(
                f"vqa-{split}-{question['question_id']}",
                f"{str(question['question']).strip()} Answer with a short answer only.",
                image_dir / image_name,
                answer,
                "vqa",
                image_id,
                split,
                metadata={
                    "answers": answers,
                    "answer_type": ann.get("answer_type"),
                    "question_type": ann.get("question_type"),
                },
            )
        )
    return rows


def make_example(
    task_id: str,
    question: str,
    image: Path,
    answer: Any,
    task_type: str,
    image_id: int,
    split: str,
    *,
    visible_context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "question": question,
        "image": str(image),
        "answer": answer,
        "task_type": task_type,
        "split": split,
        "visible_context": visible_context or {},
        "metadata": {"hcl_task": task_type, "image_id": image_id, **(metadata or {})},
    }


def load_coco(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
