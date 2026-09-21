# ComfyUI compatibility and GPU evidence

The custom node targets the exact ComfyUI commit recorded in
[`comfyui_support.json`](../comfyui_jev_sparse/comfyui_support.json). The manifest also records the
`comfy-kitchen` version whose symbols are required by the sparse kernel.

Validate a clean checkout before running the node:

```powershell
git clone https://github.com/comfyanonymous/ComfyUI.git C:\work\ComfyUI
git -C C:\work\ComfyUI checkout c194dd00cd42aa18d9dbf27d977bf6b85d9ea565
python tools/check_comfyui_compatibility.py --comfyui-path C:\work\ComfyUI
```

The compatibility command verifies the exact Git revision, imports every internal module and symbol used by
the node, and imports `comfyui_jev_sparse` so node registration failures are visible. CI runs the manifest-only
check; it does not pretend that a hosted CPU runner replaces a ComfyUI/CUDA integration check.

## GPU workflow evidence

Issue #4 is not complete until the following three workflows run on the same clean ComfyUI checkout, model,
seed, resolution, and sampling settings:

1. dense H3 baseline;
2. fixed sparse baseline;
3. Jev-guided sparse path.

Record the tested custom-node commit, ComfyUI revision, `comfy-kitchen` version, GPU model/VRAM, driver and
CUDA runtime, resolution, steps, seed, wall-clock time, peak VRAM, planner latency, selected ratios, workflow
success/failure, and output-quality comparison. A real CUDA workflow is intentionally an external gate; this
repository does not claim that the reference speedup or quality equivalence has been reproduced without that
record.

Planner failures use the configured conservative ratio but always emit a warning with the step and underlying
error. This is a deliberate safety fallback, not a silent success claim.
