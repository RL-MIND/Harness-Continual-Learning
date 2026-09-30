from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, SiglipModel


class CapabilityModelService:
    def __init__(
        self,
        *,
        bge_model_path: str,
        siglip_model_path: str,
        corpus_paths: list[str],
        corpus_id: str,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        self.device = torch.device(device)
        self.corpus_id = corpus_id
        self.batch_size = max(1, batch_size)
        self.max_length = max(32, max_length)
        self.bge_model_path = str(Path(bge_model_path).resolve())
        self.siglip_model_path = str(Path(siglip_model_path).resolve())
        self.bge_revision = _model_revision(Path(self.bge_model_path))
        self.siglip_revision = _model_revision(Path(self.siglip_model_path))
        self.documents = _load_documents([Path(item) for item in corpus_paths])
        if not self.documents:
            raise ValueError("No answer-free corpus documents were loaded.")

        self.bge_tokenizer = AutoTokenizer.from_pretrained(
            self.bge_model_path, local_files_only=True
        )
        self.bge_model = AutoModel.from_pretrained(
            self.bge_model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.bge_model.eval()
        self.document_embeddings = self._embed(
            [str(item["text"]) for item in self.documents]
        ).cpu()
        self.index_version = f"{self.corpus_id}_train_answer_free_{len(self.documents)}"

        self._siglip_lock = threading.Lock()
        self._siglip_processor: Any | None = None
        self._siglip_model: Any | None = None

    @torch.inference_mode()
    def _embed(self, texts: list[str]) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.bge_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.bge_model(**encoded)
            dense = functional.normalize(output.last_hidden_state[:, 0], p=2, dim=-1)
            chunks.append(dense)
        return torch.cat(chunks, dim=0)

    def semantic_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        query = str(payload.get("query", "")).strip()
        corpus_id = str(payload.get("corpus_id", "")).strip()
        top_k = int(payload.get("top_k", 5))
        if not query or len(query) > 8000:
            raise ValueError("query must contain 1 to 8000 characters")
        if corpus_id != self.corpus_id:
            raise ValueError(f"Unknown corpus_id: {corpus_id}")
        top_k = max(1, min(top_k, 20, len(self.documents)))
        query_embedding = self._embed([query]).cpu()[0]
        scores = torch.mv(self.document_embeddings, query_embedding)
        values, indices = torch.topk(scores, k=top_k)
        hits: list[dict[str, Any]] = []
        for score, index in zip(values.tolist(), indices.tolist()):
            document = self.documents[index]
            hits.append(
                {
                    "item_id": document["item_id"],
                    "score": float(score),
                    "text": document["text"],
                    "metadata": document["metadata"],
                }
            )
        return {
            "hits": hits,
            "model": "BAAI/bge-m3",
            "model_revision": self.bge_revision,
            "index_version": self.index_version,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _load_siglip(self) -> tuple[Any, Any]:
        with self._siglip_lock:
            if self._siglip_model is None:
                self._siglip_processor = AutoProcessor.from_pretrained(
                    self.siglip_model_path, local_files_only=True
                )
                self._siglip_model = SiglipModel.from_pretrained(
                    self.siglip_model_path,
                    local_files_only=True,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                ).to(self.device)
                self._siglip_model.eval()
        return self._siglip_processor, self._siglip_model

    @torch.inference_mode()
    def cross_modal_match(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        attachment = payload.get("attachment")
        candidates = payload.get("candidates")
        top_k = int(payload.get("top_k", 5))
        if not isinstance(attachment, dict) or attachment.get("type") != "image":
            raise ValueError("attachment must be a task-authorized image")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates must be a non-empty list")
        image_path = Path(str(attachment.get("path", ""))).resolve()
        if not image_path.is_file():
            raise ValueError("authorized image does not exist")
        clean_candidates = [
            {"id": str(item["id"]), "text": str(item["text"])}
            for item in candidates[:100]
            if isinstance(item, dict) and item.get("id") and item.get("text")
        ]
        if not clean_candidates:
            raise ValueError("no valid text candidates")
        processor, model = self._load_siglip()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            inputs = processor(
                text=[item["text"] for item in clean_candidates],
                images=image,
                padding="max_length",
                return_tensors="pt",
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = model(**inputs).logits_per_image[0].float().cpu()
        order = torch.argsort(logits, descending=True).tolist()
        top_k = max(1, min(top_k, 20, len(clean_candidates)))
        matches = [
            {
                "candidate_id": clean_candidates[index]["id"],
                "score": float(logits[index]),
                "rank": rank,
            }
            for rank, index in enumerate(order[:top_k], start=1)
        ]
        return {
            "matches": matches,
            "model": "google/siglip2-base-patch16-224",
            "model_revision": self.siglip_revision,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def _load_documents(paths: list[Path]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                task_name = str(record.get("task_name") or path.parent.name)
                source_id = str(record.get("id") or f"line-{line_number}")
                visible_context = record.get("visible_context")
                if isinstance(visible_context, list):
                    for context_index, item in enumerate(visible_context):
                        if not isinstance(item, dict):
                            continue
                        title = str(item.get("title", "")).strip()
                        paragraph = str(item.get("paragraph_text", "")).strip()
                        text = f"[{title}] {paragraph}".strip()
                        _append_document(
                            documents,
                            seen,
                            item_id=f"{task_name}:{source_id}:context-{context_index}",
                            text=text,
                            metadata={"task_name": task_name, "source_id": source_id, "kind": "context"},
                        )
                question = str(record.get("question", "")).strip()
                _append_document(
                    documents,
                    seen,
                    item_id=f"{task_name}:{source_id}:question",
                    text=question,
                    metadata={"task_name": task_name, "source_id": source_id, "kind": "question"},
                )
    return documents


def _append_document(
    documents: list[dict[str, Any]],
    seen: set[str],
    *,
    item_id: str,
    text: str,
    metadata: dict[str, str],
) -> None:
    text = text.strip()
    if not text or text in seen:
        return
    seen.add(text)
    documents.append({"item_id": item_id, "text": text[:12000], "metadata": metadata})


def _model_revision(model_path: Path) -> str:
    commit_file = model_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if commit_file.exists():
        parts = commit_file.read_text(encoding="utf-8").strip().splitlines()
        if parts:
            return parts[-1]
    return f"local-{model_path.name}"


def _handler(service: CapabilityModelService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._respond(404, {"error": "not_found"})
                return
            self._respond(
                200,
                {
                    "status": "ok",
                    "device": str(service.device),
                    "corpus_id": service.corpus_id,
                    "documents": len(service.documents),
                    "index_version": service.index_version,
                },
            )

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                if self.path == "/v1/text/search":
                    result = service.semantic_search(payload)
                elif self.path == "/v1/multimodal/match":
                    result = service.cross_modal_match(payload)
                else:
                    self._respond(404, {"error": "not_found"})
                    return
                self._respond(200, result)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self._respond(400, {"error": type(exc).__name__, "detail": str(exc)[:500]})
            except Exception as exc:
                self._respond(500, {"error": type(exc).__name__, "detail": str(exc)[:500]})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[capability-service] {self.address_string()} {format % args}", flush=True)

        def _respond(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="HCL external model capability service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bge-model", required=True)
    parser.add_argument("--siglip-model", required=True)
    parser.add_argument("--corpus", action="append", required=True)
    parser.add_argument("--corpus-id", default="reasoning_train_corpus")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    service = CapabilityModelService(
        bge_model_path=args.bge_model,
        siglip_model_path=args.siglip_model,
        corpus_paths=args.corpus,
        corpus_id=args.corpus_id,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    server = ThreadingHTTPServer((args.host, args.port), _handler(service))
    print(
        json.dumps(
            {
                "event": "ready",
                "host": args.host,
                "port": args.port,
                "device": args.device,
                "documents": len(service.documents),
                "index_version": service.index_version,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
