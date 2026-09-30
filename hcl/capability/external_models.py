from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


class ExternalCapabilityServiceError(RuntimeError):
    """Raised when an external capability service cannot satisfy a request."""


@dataclass(frozen=True)
class ExternalModelServiceClient:
    """Thin HTTP client for model-backed capabilities.

    The client deliberately does not import transformers, load weights, perform a
    health check, or start a model process. BGE-M3 and SigLIP 2 can therefore be
    deployed later without changing the HCL Router/capability contract.
    """

    base_url: str
    timeout_seconds: float = 15.0

    def semantic_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/v1/text/search", payload)

    def cross_modal_match(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/v1/multimodal/match", payload)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        base_url = self.base_url.strip().rstrip("/")
        if not base_url:
            raise ExternalCapabilityServiceError("External capability service URL is not configured.")
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            f"{base_url}{path}",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=max(float(self.timeout_seconds), 0.1)) as response:
                body = response.read().decode("utf-8")
        except (error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
            raise ExternalCapabilityServiceError(f"External capability request failed: {exc}") from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ExternalCapabilityServiceError("External capability service returned invalid JSON.") from exc
        if not isinstance(decoded, dict):
            raise ExternalCapabilityServiceError("External capability service must return a JSON object.")
        return decoded
