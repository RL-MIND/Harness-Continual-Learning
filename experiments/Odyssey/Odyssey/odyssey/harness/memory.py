from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any

import odyssey.utils as U

from .common import now_iso


MEMORY_TYPES = ("lessons", "task_strategies", "failure_patterns", "skill_notes")
SCHEMA_VERSION = 2
_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*|[0-9]+|[\u4e00-\u9fff]")
_ITEM_ID = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_SKILL_NAME = re.compile(r"\b[a-z]+(?:[A-Z][a-zA-Z0-9]*)+\b")


class ExperienceMemory:
    """Persistent episodic log plus structured, hybrid-retrievable memory.

    The public retrieval API intentionally still returns strings: the Harness
    prompts should not need to understand the storage schema.  On disk each
    abstract memory is a record with provenance, confidence and lifecycle
    metadata.  Existing version-1 files containing lists of strings are
    migrated in place on first load.
    """

    def __init__(
        self,
        ckpt_dir: str = "ckpt",
        resume: bool = True,
        embedding_model: Any | None = None,
        backend_config: str | None = None,
    ):
        self.root = U.f_mkdir(ckpt_dir, "harness")
        self.raw_path = U.f_join(self.root, "raw_memory.jsonl")
        self.abstract_path = U.f_join(self.root, "abstract_memory.json")
        self.embedding_cache_path = U.f_join(self.root, "abstract_memory_embeddings.json")
        self.proposed_updates_path = U.f_join(self.root, "proposed_updates.jsonl")
        self.embedding_model = embedding_model
        self.backend_config_path = backend_config or os.getenv("ODYSSEY_MEMORY_CONFIG", "")
        self.backend_config = self._load_backend_config(self.backend_config_path)
        backend = str(self.backend_config.get("backend", "hybrid")).lower()
        if backend not in {"hybrid", "retry"}:
            backend = "hybrid"
        self.backend = backend
        self.retry_only = backend == "retry"
        raw_updates = self.backend_config.get("updates") or {}
        if not isinstance(raw_updates, dict):
            raw_updates = {}
        default_non_memory = backend == "hybrid"
        non_memory = self._config_bool(raw_updates, "non_memory", default_non_memory)
        self.update_policy = {
            "memory": self._config_bool(raw_updates, "memory", not self.retry_only),
            "non_memory": non_memory,
            "skills": self._config_bool(raw_updates, "skills", non_memory),
            "prompts": self._config_bool(raw_updates, "prompts", non_memory),
            "skill_health": self._config_bool(raw_updates, "skill_health", non_memory),
        }
        self.memory_updates_enabled = self.update_policy["memory"]
        self.skill_updates_enabled = self.update_policy["skills"]
        self.prompt_updates_enabled = self.update_policy["prompts"]
        self.skill_health_updates_enabled = self.update_policy["skill_health"]
        raw_execution = self.backend_config.get("execution") or {}
        if not isinstance(raw_execution, dict):
            raw_execution = {}
        self.retry_suppression_enabled = self._config_bool(
            raw_execution,
            "retry_suppression",
            not self.retry_only,
        )
        self.condition_on_failure_feedback = self._config_bool(
            raw_execution,
            "condition_on_failure_feedback",
            not self.retry_only,
        )
        self.use_harness_learned_skills = self._config_bool(
            raw_execution,
            "use_harness_learned_skills",
            not self.retry_only,
        )
        self._embedding_failed = False
        if not resume:
            U.f_remove(self.raw_path)
            U.f_remove(self.abstract_path)
            U.f_remove(self.embedding_cache_path)
            U.f_remove(self.proposed_updates_path)
        self.abstract_memory = (
            U.load_json(self.abstract_path) if U.f_exists(self.abstract_path) else self._empty_store()
        )
        changed = self._migrate_store()
        self.embedding_cache = self._load_embedding_cache()
        if changed or not U.f_exists(self.abstract_path):
            self.save_abstract()

    def _empty_store(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            **{memory_type: [] for memory_type in MEMORY_TYPES},
            "updated_at": None,
        }

    def append_raw(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {"time": now_iso(), **record}
        # Retry-only is a zero-learning baseline; its execution remains in the
        # curriculum/telemetry logs but must not enter the memory pipeline.
        if not self.retry_only:
            with open(self.raw_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def append_proposed_update(self, update: dict[str, Any]) -> None:
        update = {"time": now_iso(), **update}
        with open(self.proposed_updates_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(update, ensure_ascii=False) + "\n")

    def save_abstract(self) -> None:
        U.f_mkdir_in_path(self.abstract_path)
        with open(self.abstract_path, "w", encoding="utf-8") as fp:
            json.dump(self.abstract_memory, fp, ensure_ascii=False, indent=2)

    def merge_abstract(
        self,
        update: dict[str, Any],
        *,
        source_raw_time: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Merge approved memories, consolidating exact normalized duplicates."""
        changed = False
        for memory_type in MEMORY_TYPES:
            incoming = update.get(memory_type, [])
            if isinstance(incoming, (str, dict)):
                incoming = [incoming]
            if not isinstance(incoming, list):
                continue
            existing = self.abstract_memory.setdefault(memory_type, [])
            by_content = {
                self._normalize_content(record.get("content", "")): record
                for record in existing
                if isinstance(record, dict)
            }
            for item in incoming:
                record = self._coerce_record(
                    item,
                    memory_type,
                    source_raw_time=source_raw_time,
                    context=context,
                )
                if record is None:
                    continue
                key = self._normalize_content(record["content"])
                duplicate = by_content.get(key)
                if duplicate is not None:
                    duplicate["evidence_count"] = int(duplicate.get("evidence_count", 1)) + 1
                    duplicate["confidence"] = round(
                        min(0.99, max(float(duplicate.get("confidence", 0.6)), float(record["confidence"])) + 0.03),
                        3,
                    )
                    duplicate["updated_at"] = now_iso()
                    self._merge_unique(duplicate, record, "entities")
                    self._merge_unique(duplicate, record, "skills")
                    self._merge_unique(duplicate, record, "source_raw_times")
                    changed = True
                    continue
                existing.append(record)
                by_content[key] = record
                changed = True
        if changed:
            self.abstract_memory["schema_version"] = SCHEMA_VERSION
            self.abstract_memory["updated_at"] = now_iso()
            self.save_abstract()

    def retrieve_abstract(self, query: str, limit: int = 12) -> list[str]:
        """Retrieve memories using semantic, lexical and structured signals.

        Embeddings are optional.  Any model or endpoint failure disables the
        semantic component for this process and cleanly falls back to local
        scoring, so memory cannot make an agent task unavailable.
        """
        if self.retry_only:
            return []
        records = self._active_records()
        if not records or limit <= 0:
            return []

        query_tokens = self._tokens(query)
        query_entities = self._extract_entities(query)
        query_skills = self._extract_skills(query)
        semantic_scores = self._semantic_scores(query, records)
        now = datetime.now(timezone.utc)
        ranked: list[tuple[float, float, dict[str, Any]]] = []
        for record in records:
            content = self._normalize_equipment_memory(str(record.get("content", "")))
            content_tokens = self._tokens(content)
            lexical = self._jaccard(query_tokens, content_tokens)
            entity = self._overlap(query_entities, set(record.get("entities", [])))
            skill = self._overlap(query_skills, set(record.get("skills", [])))
            confidence = self._clamp(float(record.get("confidence", 0.6)))
            recency = self._recency_score(record.get("updated_at") or record.get("created_at"), now)
            evidence = min(1.0, math.log1p(max(1, int(record.get("evidence_count", 1)))) / math.log(6))
            semantic = semantic_scores.get(record["id"], 0.0)
            if semantic_scores:
                score = (
                    0.45 * semantic + 0.20 * lexical + 0.12 * entity
                    + 0.08 * skill + 0.08 * confidence + 0.04 * recency + 0.03 * evidence
                )
            else:
                # Re-normalized local-only weights; unlike the old strategy,
                # this handles Chinese via character tokens and favors recent,
                # repeatedly evidenced records when no lexical term matches.
                score = (
                    0.45 * lexical + 0.20 * entity + 0.12 * skill
                    + 0.12 * confidence + 0.06 * recency + 0.05 * evidence
                )
            ranked.append((score, self._timestamp(record), record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [record for _, _, record in ranked[: min(limit, len(ranked))]]
        retrieval_time = now_iso()
        for record in selected:
            record["last_retrieved_at"] = retrieval_time
            record["retrieval_count"] = int(record.get("retrieval_count", 0)) + 1
        self.save_abstract()
        return [f"{record['type']}: {self._normalize_equipment_memory(record['content'])}" for record in selected]

    def _load_backend_config(self, path: str) -> dict[str, Any]:
        if not path:
            return {}
        try:
            with open(path, encoding="utf-8") as fp:
                value = json.load(fp)
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
        value = config.get(key)
        return value if isinstance(value, bool) else default

    def _migrate_store(self) -> bool:
        if not isinstance(self.abstract_memory, dict):
            self.abstract_memory = self._empty_store()
            return True
        changed = self.abstract_memory.get("schema_version") != SCHEMA_VERSION
        for memory_type in MEMORY_TYPES:
            values = self.abstract_memory.get(memory_type, [])
            if not isinstance(values, list):
                values = [values] if values else []
                changed = True
            migrated = []
            for item in values:
                record = self._coerce_record(item, memory_type)
                if record is not None:
                    migrated.append(record)
                if not isinstance(item, dict) or item != record:
                    changed = True
            self.abstract_memory[memory_type] = migrated
        self.abstract_memory["schema_version"] = SCHEMA_VERSION
        self.abstract_memory.setdefault("updated_at", now_iso() if changed else None)
        return changed

    def _coerce_record(
        self,
        item: Any,
        memory_type: str,
        *,
        source_raw_time: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if isinstance(item, str):
            content = item.strip()
            supplied: dict[str, Any] = {}
        elif isinstance(item, dict):
            content = str(item.get("content") or item.get("text") or "").strip()
            supplied = item
        else:
            return None
        if not content:
            return None
        context = context or {}
        created_at = str(supplied.get("created_at") or now_iso())
        sources = self._string_list(supplied.get("source_raw_times", []))
        if source_raw_time and source_raw_time not in sources:
            sources.append(source_raw_time)
        record_id = str(supplied.get("id") or self._memory_id(memory_type, content))
        record = {
            "id": record_id,
            "type": memory_type,
            "content": content,
            "task_kind": str(supplied.get("task_kind") or context.get("task_kind") or "unknown"),
            "entities": self._string_list(supplied.get("entities")) or sorted(self._extract_entities(content)),
            "skills": self._string_list(supplied.get("skills"))
            or self._string_list(context.get("skills"))
            or sorted(self._extract_skills(content)),
            "conditions": self._string_list(supplied.get("conditions"))
            or self._string_list(context.get("conditions")),
            "outcome": str(
                supplied.get("outcome") or context.get("outcome") or self._infer_outcome(memory_type)
            ),
            "confidence": round(self._clamp(self._number(supplied.get("confidence"), 0.6)), 3),
            "evidence_count": max(1, self._integer(supplied.get("evidence_count"), 1)),
            "source_raw_times": sources,
            "created_at": created_at,
            "updated_at": str(supplied.get("updated_at") or created_at),
            "last_retrieved_at": supplied.get("last_retrieved_at"),
            "retrieval_count": max(0, self._integer(supplied.get("retrieval_count"), 0)),
            "status": str(supplied.get("status", "active")),
        }
        return record

    def _semantic_scores(self, query: str, records: list[dict[str, Any]]) -> dict[str, float]:
        if self.embedding_model is None or self._embedding_failed:
            return {}
        try:
            missing = [
                record
                for record in records
                if self.embedding_cache.get(record["id"], {}).get("content_hash")
                != self._content_hash(record["content"])
            ]
            if missing:
                vectors = self.embedding_model.embed_documents([record["content"] for record in missing])
                if len(vectors) != len(missing):
                    raise ValueError("embedding count does not match memory count")
                for record, vector in zip(missing, vectors):
                    self.embedding_cache[record["id"]] = {
                        "content_hash": self._content_hash(record["content"]),
                        "vector": vector,
                    }
                self._save_embedding_cache()
            query_vector = self.embedding_model.embed_query(query)
            scores = {}
            for record in records:
                cached = self.embedding_cache.get(record["id"], {})
                if cached.get("content_hash") != self._content_hash(record["content"]):
                    continue
                scores[record["id"]] = max(0.0, self._cosine(query_vector, cached.get("vector", [])))
            return scores
        except Exception:  # Embedding is an optional retrieval enhancement.
            self._embedding_failed = True
            return {}

    def _load_embedding_cache(self) -> dict[str, dict[str, Any]]:
        if not U.f_exists(self.embedding_cache_path):
            return {}
        try:
            value = U.load_json(self.embedding_cache_path)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _save_embedding_cache(self) -> None:
        with open(self.embedding_cache_path, "w", encoding="utf-8") as fp:
            json.dump(self.embedding_cache, fp, ensure_ascii=False)

    def _active_records(self) -> list[dict[str, Any]]:
        return [
            record
            for memory_type in MEMORY_TYPES
            for record in self.abstract_memory.get(memory_type, [])
            if isinstance(record, dict) and record.get("status", "active") == "active"
        ]

    def _tokens(self, text: str) -> set[str]:
        raw = _WORD.findall(str(text).lower().replace("_", " "))
        tokens = {token for token in raw if len(token) > 1 or "\u4e00" <= token <= "\u9fff"}
        chinese = "".join(token for token in raw if len(token) == 1 and "\u4e00" <= token <= "\u9fff")
        tokens.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
        return tokens

    def _extract_entities(self, text: str) -> set[str]:
        return {match.lower() for match in _ITEM_ID.findall(str(text))}

    def _extract_skills(self, text: str) -> set[str]:
        return set(_SKILL_NAME.findall(str(text)))

    def _memory_id(self, memory_type: str, content: str) -> str:
        digest = hashlib.sha256(f"{memory_type}\0{self._normalize_content(content)}".encode()).hexdigest()[:16]
        return f"mem_{digest}"

    def _content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def _normalize_content(self, content: str) -> str:
        return " ".join(str(content).lower().split())

    def _infer_outcome(self, memory_type: str) -> str:
        return "failure" if memory_type == "failure_patterns" else "unknown"

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return list(dict.fromkeys(str(item) for item in value if str(item).strip()))

    def _merge_unique(self, target: dict[str, Any], source: dict[str, Any], key: str) -> None:
        target[key] = list(dict.fromkeys(self._string_list(target.get(key)) + self._string_list(source.get(key))))

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        return len(left & right) / len(left | right) if left and right else 0.0

    def _overlap(self, left: set[str], right: set[str]) -> float:
        return len(left & right) / len(left) if left else 0.0

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
        return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _number(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _integer(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _recency_score(self, value: Any, now: datetime) -> float:
        try:
            parsed = datetime.fromisoformat(str(value))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            days = max(0.0, (now - parsed.astimezone(timezone.utc)).total_seconds() / 86400)
            return math.exp(-days / 90)
        except (TypeError, ValueError):
            return 0.0

    def _timestamp(self, record: dict[str, Any]) -> float:
        try:
            return datetime.fromisoformat(str(record.get("updated_at") or record.get("created_at"))).timestamp()
        except (TypeError, ValueError):
            return 0.0

    def _normalize_equipment_memory(self, item: str) -> str:
        replacements = {
            "equipment[4] (off-hand)": "equipment[4] (main_hand)",
            "equipment[4] - off-hand": "equipment[4] - main_hand",
            "equipment[4], off-hand": "equipment[4], main_hand",
            "equipment[4] is off-hand": "equipment[4] is main_hand",
            "equipment[4] is the off-hand": "equipment[4] is the main_hand",
            "slot 4 (off-hand)": "slot 4 (main_hand)",
            "slot [4] and off-hand": "slot [4] and main_hand",
            "slot [4] which is likely not the off-hand": "slot [4] which is main_hand",
        }
        normalized = item
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized
