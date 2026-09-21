# Project-specific Codex instructions

## Project purpose

- Develop and validate the Jev-guided MiniMax H3 sparse-attention prototype spanning a Cloudflare Worker planner and a ComfyUI custom node.

## Project-specific sources of truth

- `README.md`: architecture, supported execution paths, known limitations, and benchmark requirements.
- `package.json`: Worker development/test commands.
- `wrangler.*` / Worker source: deployed planner contract.
- `comfyui_jev_sparse/`: ComfyUI integration boundary.

## Project-specific invariants

- Do not claim the reported sparse-attention speedup or quality equivalence without repository-owned benchmark evidence under identical model/seed/resolution/sampling conditions.
- Keep the direct `/plan` path and MCP path on the same planner semantics; do not create divergent policy logic.
- Do not send Q/K/V tensors or other large private model-state payloads to Cloudflare. Only the documented compact step context/optional sampled statistics may cross that boundary.
- Treat ComfyUI internal sparse-attention APIs as compatibility-sensitive; pin or validate the supported upstream revision before changing those integration points.
- Never commit Worker tokens or other secrets.

## Project verification

- Worker checks: `pnpm run check`.
- Python/custom-node syntax smoke: `python -m compileall -q comfyui_jev_sparse`.
- Performance/quality claims require dense, fixed-sparse, and Jev-guided runs under matched conditions.
- Public/non-loopback Worker requests to `/plan` and `/mcp` must fail closed when `SHARED_TOKEN` is absent. Unauthenticated use is allowed only by an explicit local-development opt-in on a loopback host.
