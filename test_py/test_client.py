from __future__ import annotations

import json
import os
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from comfyui_jev_sparse.client import PlannerClient, PlannerError


class FakeResponse:
    def __init__(self, payload: object = None, *, raw: bytes | None = None) -> None:
        self._raw = raw if raw is not None else json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._raw


class RecordingOpener:
    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.request: urllib.request.Request | None = None
        self.timeout: float | None = None

    def __call__(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeResponse:
        self.request = request
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class PlannerClientTests(unittest.TestCase):
    def test_plan_uses_injected_transport_and_builds_request(self) -> None:
        opener = RecordingOpener(
            FakeResponse(
                {
                    "keep_percent": [10, 5],
                    "confidence": [0.9, 0.8],
                    "fallback_layers": [],
                }
            )
        )
        client = PlannerClient("https://example.test/mcp", timeout_s=1.25, opener=opener)

        with patch.dict(os.environ, {"JEV_SPARSE_TOKEN": "secret"}, clear=False):
            result = client.plan(
                step=3,
                sigma=0.7,
                sequence_length=4096,
                layer_count=2,
                fallback_keep_percent=10,
                min_confidence=0.55,
                policy="balanced",
                layer_metrics=[{"layer": 0, "activation_rms": 0.4}],
            )

        self.assertEqual(result["keep_percent"], [10, 5])
        self.assertIsNotNone(opener.request)
        assert opener.request is not None
        self.assertEqual(opener.request.full_url, "https://example.test/plan")
        self.assertEqual(opener.request.get_method(), "POST")
        self.assertEqual(opener.request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(opener.timeout, 1.25)
        payload = json.loads((opener.request.data or b"{}").decode("utf-8"))
        self.assertEqual(payload["step"], 3)
        self.assertEqual(payload["layer_count"], 2)
        self.assertEqual(payload["allowed_keep_percent"], [1, 3, 5, 10])
        self.assertEqual(payload["sequence_length"], 4096)

    def test_plan_maps_network_failure_to_planner_error(self) -> None:
        opener = RecordingOpener(error=urllib.error.URLError("offline"))
        client = PlannerClient("https://example.test/plan", opener=opener)

        with self.assertRaisesRegex(PlannerError, "planner request failed"):
            client.plan(
                step=1,
                sigma=None,
                sequence_length=None,
                layer_count=1,
                fallback_keep_percent=10,
                min_confidence=0.55,
                policy="quality",
                layer_metrics=[],
            )

    def test_plan_rejects_malformed_json_without_network(self) -> None:
        client = PlannerClient(
            "https://example.test/plan",
            opener=RecordingOpener(FakeResponse(raw=b"not-json")),
        )

        with self.assertRaisesRegex(PlannerError, "planner request failed"):
            client.plan(
                step=1,
                sigma=None,
                sequence_length=None,
                layer_count=1,
                fallback_keep_percent=10,
                min_confidence=0.55,
                policy="quality",
                layer_metrics=[],
            )

    def test_plan_rejects_invalid_layer_plan(self) -> None:
        client = PlannerClient(
            "https://example.test/plan",
            opener=RecordingOpener(FakeResponse({"keep_percent": [7]})),
        )

        with self.assertRaisesRegex(PlannerError, "unsupported keep percentage"):
            client.plan(
                step=1,
                sigma=None,
                sequence_length=None,
                layer_count=1,
                fallback_keep_percent=10,
                min_confidence=0.55,
                policy="quality",
                layer_metrics=[],
            )


if __name__ == "__main__":
    unittest.main()
