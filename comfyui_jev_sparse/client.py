from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol


class PlannerError(RuntimeError):
    pass


class ResponseLike(Protocol):
    def __enter__(self) -> "ResponseLike": ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object: ...

    def read(self) -> bytes: ...


class UrlOpener(Protocol):
    def __call__(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> ResponseLike: ...


class PlannerClient:
    def __init__(
        self,
        endpoint: str,
        timeout_s: float = 0.8,
        *,
        opener: UrlOpener | None = None,
    ) -> None:
        endpoint = endpoint.strip().rstrip("/")
        if endpoint.endswith("/mcp"):
            endpoint = endpoint[:-4] + "/plan"
        elif not endpoint.endswith("/plan"):
            endpoint += "/plan"
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._opener: UrlOpener = opener or urllib.request.urlopen

    def plan(
        self,
        *,
        step: int,
        sigma: float | None,
        sequence_length: int | None,
        layer_count: int,
        fallback_keep_percent: int,
        min_confidence: float,
        policy: str,
        layer_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step": step,
            "layer_count": layer_count,
            "allowed_keep_percent": [1, 3, 5, 10],
            "fallback_keep_percent": fallback_keep_percent,
            "min_confidence": min_confidence,
            "policy": policy,
            "layer_metrics": layer_metrics,
        }
        if sigma is not None:
            payload["sigma"] = sigma
        if sequence_length is not None and sequence_length > 0:
            payload["sequence_length"] = sequence_length

        headers = {"content-type": "application/json"}
        token = os.getenv("JEV_SPARSE_TOKEN")
        if token:
            headers["authorization"] = f"Bearer {token}"

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with self._opener(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PlannerError(f"planner request failed: {exc}") from exc

        keep = data.get("keep_percent")
        if not isinstance(keep, list) or len(keep) != layer_count:
            raise PlannerError("planner returned an invalid layer plan")
        if any(value not in (1, 3, 5, 10) for value in keep):
            raise PlannerError("planner returned an unsupported keep percentage")
        return data
