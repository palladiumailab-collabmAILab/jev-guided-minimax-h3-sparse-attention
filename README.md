# Jev-guided MiniMax H3 sparse attention

Experimental prototype for driving MiniMax H3 block-sparse attention with TypeSafe Jev on Cloudflare Workers AI.

## Architecture

```text
ComfyUI / RTX GPU
  MiniMax H3
      |
      | once per denoising step
      v
Cloudflare Worker
  /plan  ----+
  /mcp       | shared planner
             v
      Workers AI: typesafe/jev
             |
             v
      per-layer keep %
      [1, 3, 5, 10]
             |
             v
ComfyUI sparse kernel
  comfy_kitchen.sol_attn_chunked
```

The MCP endpoint exists for agent/tool use. ComfyUI uses the direct `/plan` hot path to avoid MCP handshake overhead during generation. Both paths call the same planner implementation.

No Q/K/V tensors are sent to Cloudflare. The request contains denoising-step context and, optionally, small sampled activation statistics from the previous step.

## What is implemented

- Stateless Cloudflare MCP server at `/mcp` using MCP SDK v2 `createMcpHandler`.
- Low-overhead planner endpoint at `/plan`.
- Cloudflare Workers AI model `typesafe/jev`.
- One Jev inference per denoising step.
- One typed Jev Choice decision per H3 transformer block.
- Allowed keep ratios: 1%, 3%, 5%, 10%.
- Confidence-gated fallback to a configured conservative ratio.
- MiniMax H3 ComfyUI custom node.
- Per-block `topk_ratio` injected into ComfyUI's existing H3 `sol_attn_chunked` path.
- Optional sampled activation RMS/variance telemetry.
- Bearer-token protection is required for non-loopback/public requests. Missing `SHARED_TOKEN` is fail-closed;
  unauthenticated access is available only when `ALLOW_INSECURE_LOCAL_DEV=true` and the request host is loopback.

## Important limitation

This repository implements the mechanism. It does **not** establish that the 41.7% speedup shown in the reference post is reproducible, nor that 1/3/5/10% top-k attention is quality-equivalent to dense attention.

ComfyUI currently documents fixed-ratio SLA-style sparsity as an experimental path; aggressive ratios can change output quality unless the model/weights are suitable for that sparse pattern. Benchmark dense, fixed sparse, and Jev-guided runs with identical model, seed, resolution, and sampling settings.

The custom node currently depends on ComfyUI's experimental MiniMax H3 sparse-attention internals, including `comfy_extras.nodes_sparse_attention` and `comfy_kitchen.sol_attn_chunked`. Upstream API changes may require adaptation.

## Deploy the Worker

Requirements: Node.js, pnpm, a Cloudflare account with Workers AI enabled.

```bash
corepack enable
corepack prepare pnpm@10.15.1 --activate
pnpm install
pnpm exec wrangler login
pnpm run deploy
```

Required before a public deployment:

```bash
pnpm exec wrangler secret put SHARED_TOKEN
```

Without `SHARED_TOKEN`, `/plan` and `/mcp` return `401` for deployed or non-loopback hosts. For local-only
development, opt in explicitly:

```bash
pnpm run dev -- --var ALLOW_INSECURE_LOCAL_DEV:true
```

The opt-in is still rejected when the request host is not `localhost`, `127.0.0.1`, or `::1`.

The Worker exposes:

- `GET /health`
- `POST /plan`
- `POST /mcp`

Local development:

```bash
pnpm run dev
```

## Install the ComfyUI node

Place or symlink `comfyui_jev_sparse` into `ComfyUI/custom_nodes/`, then restart ComfyUI.

If `SHARED_TOKEN` is configured on the Worker, set the same value in the ComfyUI process environment as `JEV_SPARSE_TOKEN`.

Example node settings:

```text
endpoint: https://<worker>.workers.dev/plan
start_percent: 0.2
end_percent: 1.0
fallback_keep_percent: 10
policy: balanced
min_confidence: 0.55
collect_metrics: false
```

`collect_metrics=false` is the default because reading sampled GPU statistics back to CPU introduces synchronization overhead. When enabled, the next Jev request receives sampled activation RMS, variance, and inter-step RMS change for each eligible layer.

## Planner request

```json
{
  "step": 1,
  "sigma": 0.73,
  "sequence_length": 18432,
  "layer_count": 49,
  "allowed_keep_percent": [1, 3, 5, 10],
  "fallback_keep_percent": 10,
  "min_confidence": 0.55,
  "policy": "balanced",
  "layer_metrics": []
}
```

Response:

```json
{
  "keep_percent": [10, 5, 5, 3],
  "confidence": [0.91, 0.83, 0.79, 0.74],
  "fallback_layers": [],
  "average_keep_percent": 5.75,
  "model": "jev-1.x",
  "usage": {}
}
```

The real array length equals the H3 block count.

## Verification

```bash
pnpm run check
python -m compileall -q comfyui_jev_sparse
```

For performance validation, compare at minimum:

1. dense H3 baseline;
2. fixed 10% sparse baseline;
3. Jev-guided sparse run.

Record wall-clock generation time, peak VRAM, planner latency, selected ratios, and output-quality comparisons under identical generation settings.

## Tracking

Implementation plan: issue #1.
