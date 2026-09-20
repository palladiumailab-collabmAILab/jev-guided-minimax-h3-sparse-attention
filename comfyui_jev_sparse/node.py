from __future__ import annotations

from .client import PlannerClient
from .patch import apply_jev_sparse_attention


class JevMiniMaxH3SparseAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "endpoint": (
                    "STRING",
                    {"default": "http://127.0.0.1:8787/plan"},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "fallback_keep_percent": (["1", "3", "5", "10"], {"default": "10"}),
                "policy": (["speed", "balanced", "quality"], {"default": "balanced"}),
                "min_confidence": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "timeout_ms": (
                    "INT",
                    {"default": 800, "min": 100, "max": 5000, "step": 100},
                ),
                "min_tokens": (
                    "INT",
                    {"default": 12288, "min": 0, "max": 1048576, "step": 512},
                ),
                "dense_blocks": ("STRING", {"default": ""}),
                "sink_conditioning": (
                    ["exact_kv", "exact_kv_and_rows", "off"],
                    {"default": "exact_kv_and_rows"},
                ),
                "extra_tokens": (
                    "INT",
                    {"default": 0, "min": 0, "max": 256, "step": 64},
                ),
                "collect_metrics": ("BOOLEAN", {"default": False}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "model/patch"
    DESCRIPTION = (
        "Experimental MiniMax H3 patch that asks a Jev planner for a per-block "
        "1/3/5/10% sparse-attention keep ratio once per denoising step."
    )

    def patch(
        self,
        model,
        endpoint: str,
        start_percent: float,
        end_percent: float,
        fallback_keep_percent: str,
        policy: str,
        min_confidence: float,
        timeout_ms: int,
        min_tokens: int,
        dense_blocks: str,
        sink_conditioning: str,
        extra_tokens: int,
        collect_metrics: bool,
        verbose: bool,
    ):
        if end_percent < start_percent:
            raise ValueError("end_percent must be greater than or equal to start_percent")

        planner = PlannerClient(endpoint=endpoint, timeout_s=timeout_ms / 1000.0)
        patched = apply_jev_sparse_attention(
            model,
            planner=planner,
            fallback_keep_percent=int(fallback_keep_percent),
            min_confidence=min_confidence,
            policy=policy,
            collect_metrics=collect_metrics,
            start_percent=start_percent,
            end_percent=end_percent,
            min_tokens=min_tokens,
            dense_blocks=dense_blocks,
            sink_conditioning=sink_conditioning,
            extra_tokens=extra_tokens,
            verbose=verbose,
        )
        return (patched,)
