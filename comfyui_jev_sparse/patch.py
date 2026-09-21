from __future__ import annotations

import logging
import math
from typing import Any

import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy_kitchen as ck
import torch
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy_extras import nodes_sparse_attention as csa

from .client import PlannerClient, PlannerError

PRODUCER_CHUNK = getattr(csa, "PRODUCER_CHUNK", 4096)


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.reshape(-1)[0].detach().cpu().item())
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


class JevSparsePatch(csa.SparseAttnPatch):
    def __init__(
        self,
        *,
        planner: PlannerClient,
        layer_count: int,
        fallback_keep_percent: int,
        min_confidence: float,
        policy: str,
        collect_metrics: bool,
        sigma_start: float,
        sigma_end: float,
        min_tokens: int,
        dense_blocks: set[int],
        sink_conditioning: str,
        extra_tokens: int,
        verbose: bool,
    ) -> None:
        super().__init__(
            tau=1.3,
            topk_ratio=fallback_keep_percent / 100.0,
            vsa=False,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            min_tokens=min_tokens,
            dense_blocks=dense_blocks,
            sink_conditioning=sink_conditioning,
            extra_tokens=extra_tokens,
            verbose=verbose,
        )
        self.planner = planner
        self.layer_count = layer_count
        self.fallback_keep_percent = fallback_keep_percent
        self.min_confidence = min_confidence
        self.policy = policy
        self.collect_metrics = collect_metrics

        self.step_index = -1
        self.current_keep_percent = [fallback_keep_percent] * layer_count
        self.pending_metrics: dict[int, dict[str, float | int]] = {}
        self.previous_rms: dict[int, float] = {}
        self.last_tokens: int | None = None

    def keep_ratio(self, block_index: int) -> float:
        if 0 <= block_index < len(self.current_keep_percent):
            return self.current_keep_percent[block_index] / 100.0
        return self.fallback_keep_percent / 100.0

    def prepare_step(self, timestep: Any, model_options: dict[str, Any]) -> None:
        self.step_index += 1

        transformer_options = model_options.get("transformer_options", {})
        sigma = _scalar(transformer_options.get("sigmas"))
        if sigma is None:
            sigma = _scalar(timestep)

        if sigma is not None and (sigma > self.sigma_start or sigma < self.sigma_end):
            self.pending_metrics.clear()
            return

        metrics = [self.pending_metrics[key] for key in sorted(self.pending_metrics)]
        self.pending_metrics = {}

        try:
            result = self.planner.plan(
                step=self.step_index,
                sigma=sigma,
                sequence_length=self.last_tokens,
                layer_count=self.layer_count,
                fallback_keep_percent=self.fallback_keep_percent,
                min_confidence=self.min_confidence,
                policy=self.policy,
                layer_metrics=metrics,
            )
            self.current_keep_percent = [int(value) for value in result["keep_percent"]]
            self.log_once(
                ("plan", self.step_index),
                (
                    f"Jev plan step={self.step_index}, "
                    f"mean_keep={sum(self.current_keep_percent) / len(self.current_keep_percent):.2f}%"
                ),
            )
        except PlannerError as exc:
            self.current_keep_percent = [self.fallback_keep_percent] * self.layer_count
            logging.warning(
                "Jev planner unavailable at step %d; using %d%% for all layers: %s",
                self.step_index,
                self.fallback_keep_percent,
                exc,
            )

    def observe_layer(self, block_index: int, x: torch.Tensor) -> None:
        self.last_tokens = int(x.shape[0])
        if not self.collect_metrics:
            return

        rows = min(16, int(x.shape[0]))
        cols = min(64, int(x.shape[1]))
        if rows <= 0 or cols <= 0:
            return

        stride = max(1, int(x.shape[0]) // rows)
        sample = x[::stride][:rows, :cols].detach().float()
        stats = torch.stack(
            (
                torch.sqrt(torch.mean(sample * sample)),
                torch.var(sample, unbiased=False),
            )
        )
        rms, variance = (float(v) for v in stats.cpu().tolist())
        if not (math.isfinite(rms) and math.isfinite(variance)):
            return

        metric: dict[str, float | int] = {
            "layer": block_index,
            "activation_rms": rms,
            "activation_variance": variance,
        }
        previous = self.previous_rms.get(block_index)
        if previous is not None:
            metric["delta_rms"] = abs(rms - previous)
        self.previous_rms[block_index] = rms
        self.pending_metrics[block_index] = metric

    def cleanup(self) -> None:
        super().reset()
        self.step_index = -1
        self.current_keep_percent = [self.fallback_keep_percent] * self.layer_count
        self.pending_metrics.clear()
        self.previous_rms.clear()
        self.last_tokens = None


def h3_sparse_attention_dynamic(
    attn: Any,
    x: torch.Tensor,
    rope_freqs: torch.Tensor,
    transformer_options: dict[str, Any],
    patch: JevSparsePatch,
    block_index: int,
) -> torch.Tensor:
    n_tokens = x.shape[0]
    heads, head_dim = attn.heads, attn.head_dim
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)

    key = (block_index, n_tokens, tuple(transformer_options.get("uuids", ())))
    pooled = patch.pooled.get(key)
    first = pooled is None
    with comfy.model_prefetch.pause_malloc_graph():
        if first:
            pooled = (
                torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
                torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
            )

    sink, sink_q = patch.sinks(transformer_options, n_tokens)

    def chunks():
        for i in range(0, n_tokens, PRODUCER_CHUNK):
            yield attn.qkv_proj(x[i : i + PRODUCER_CHUNK])

    out, kmean, vscale = ck.sol_attn_chunked(
        chunks,
        n_tokens,
        heads,
        rope_freqs,
        (qw, kw),
        kmean=None if first else pooled[0],
        vscale=None if first else pooled[1],
        tau=patch.tau,
        topk_ratio=patch.keep_ratio(block_index),
        token_aug=patch.extra_tokens,
        sink_blocks=list(sink),
        sink_q=list(sink_q),
        rope_eps=attn.q_norm.eps,
    )
    pooled[0].copy_(kmean)
    pooled[1].copy_(vscale)
    patch.pooled[key] = pooled

    keep = patch.current_keep_percent[block_index]
    patch.log_once(
        ("producer", block_index, n_tokens, keep),
        f"block {block_index}: sparse producer, {n_tokens} tokens, keep={keep}%",
    )
    return attn.out_proj(out.view(n_tokens, heads * head_dim))


def make_h3_block_patch(block: Any, block_index: int, patch: JevSparsePatch):
    def attention(
        h: torch.Tensor,
        rope_freqs: torch.Tensor | None = None,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if rope_freqs is None:
            raise RuntimeError("MiniMax H3 sparse attention requires RoPE frequencies")
        return h3_sparse_attention_dynamic(
            block.attn,
            h,
            rope_freqs,
            transformer_options or {},
            patch,
            block_index,
        )

    def block_patch(args: dict[str, Any], extra: dict[str, Any]):
        transformer_options = args["transformer_options"]
        if csa.h3_eligible(
            block.attn,
            args["img"],
            args["rope_freqs"],
            transformer_options,
            patch,
            block_index,
        ):
            patch.observe_layer(block_index, args["img"])
            args = {**args, "attention": attention}
        return extra["original_block"](args)

    return block_patch


def apply_jev_sparse_attention(
    model: Any,
    *,
    planner: PlannerClient,
    fallback_keep_percent: int,
    min_confidence: float,
    policy: str,
    collect_metrics: bool,
    start_percent: float,
    end_percent: float,
    min_tokens: int,
    dense_blocks: str,
    sink_conditioning: str,
    extra_tokens: int,
    verbose: bool,
):
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, MiniMaxH3Model):
        raise ValueError("Jev sparse attention currently supports MiniMax H3 only")

    model_sampling = model.get_model_object("model_sampling")
    patch = JevSparsePatch(
        planner=planner,
        layer_count=len(diffusion_model.blocks),
        fallback_keep_percent=fallback_keep_percent,
        min_confidence=min_confidence,
        policy=policy,
        collect_metrics=collect_metrics,
        sigma_start=float(model_sampling.percent_to_sigma(start_percent)),
        sigma_end=float(model_sampling.percent_to_sigma(end_percent)),
        min_tokens=min_tokens,
        dense_blocks=csa.parse_block_list(dense_blocks),
        sink_conditioning=sink_conditioning,
        extra_tokens=extra_tokens,
        verbose=verbose,
    )

    patched = model.clone()
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE,
        "jev_sparse_attention",
        lambda model_patcher, timestep, model_options: patch.prepare_step(
            timestep, model_options
        ),
    )
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
        "jev_sparse_attention",
        lambda model_patcher: patch.cleanup(),
    )

    for index, block in enumerate(diffusion_model.blocks):
        patched.set_model_patch_replace(
            make_h3_block_patch(block, index, patch),
            "dit",
            "double_block",
            index,
        )

    logging.info(
        "Jev sparse attention installed for %d MiniMax H3 blocks", len(diffusion_model.blocks)
    )
    return patched
