"""Check the pinned ComfyUI source/API contract for the custom node."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path(__file__).resolve().parents[1] / "comfyui_jev_sparse" / "comfyui_support.json"


def load_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to read compatibility manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit("Compatibility manifest must be a JSON object")
    required = {"repository", "revision", "checked_at", "required_imports"}
    missing = sorted(required.difference(manifest))
    if missing:
        raise SystemExit(f"Compatibility manifest is missing fields: {', '.join(missing)}")
    revision = manifest["revision"]
    if not isinstance(revision, str) or len(revision) != 40:
        raise SystemExit("Compatibility manifest revision must be a 40-character commit SHA")
    imports = manifest["required_imports"]
    if not isinstance(imports, dict) or not imports:
        raise SystemExit("Compatibility manifest required_imports must be a non-empty object")
    for module_name, symbols in imports.items():
        if not isinstance(module_name, str) or not isinstance(symbols, list):
            raise SystemExit("Compatibility manifest import entries must map module names to symbol lists")
        if not all(isinstance(symbol, str) and symbol for symbol in symbols):
            raise SystemExit(f"Compatibility manifest has invalid symbols for {module_name}")
    return manifest


def git_output(comfyui_path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(comfyui_path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise SystemExit(f"Unable to inspect ComfyUI checkout: {detail.strip()}") from exc
    return result.stdout.strip()


def check_checkout(comfyui_path: Path, revision: str) -> None:
    if not comfyui_path.is_dir():
        raise SystemExit(f"ComfyUI path does not exist: {comfyui_path}")
    actual_revision = git_output(comfyui_path, "rev-parse", "HEAD")
    if actual_revision != revision:
        raise SystemExit(
            f"Unsupported ComfyUI revision: expected {revision}, found {actual_revision}. "
            "Check out the pinned revision before importing the node."
        )
    dirty = git_output(comfyui_path, "status", "--porcelain")
    if dirty:
        raise SystemExit("ComfyUI checkout has uncommitted changes; use a clean checkout for compatibility checks")


def check_imports(comfyui_path: Path, required_imports: dict[str, list[str]]) -> None:
    sys.path.insert(0, str(comfyui_path))
    for module_name, symbols in required_imports.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - preserve the external import failure as evidence.
            raise SystemExit(
                f"ComfyUI compatibility import failed for {module_name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        if missing:
            raise SystemExit(
                f"ComfyUI compatibility symbols missing from {module_name}: {', '.join(missing)}"
            )

    sys.path.insert(0, str(MANIFEST_PATH.parents[1]))
    try:
        importlib.import_module("comfyui_jev_sparse")
    except Exception as exc:  # noqa: BLE001 - the message is the actionable compatibility evidence.
        raise SystemExit(
            f"Custom node registration import failed: {type(exc).__name__}: {exc}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Validate the pinned manifest without requiring a local ComfyUI checkout.",
    )
    parser.add_argument(
        "--comfyui-path",
        type=Path,
        help="Clean ComfyUI checkout at the exact manifest revision; required for import checks.",
    )
    args = parser.parse_args()
    manifest = load_manifest()
    if args.manifest_only:
        print(f"Compatibility manifest valid: {manifest['revision']}")
        return 0
    if args.comfyui_path is None:
        parser.error("--comfyui-path is required unless --manifest-only is used")
    check_checkout(args.comfyui_path.resolve(), manifest["revision"])
    check_imports(args.comfyui_path.resolve(), manifest["required_imports"])
    print(f"ComfyUI compatibility imports passed at {manifest['revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
