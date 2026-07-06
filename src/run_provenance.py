"""Runtime provenance helpers for run_metadata.json (Phase 0 addition).

Captures cross-model reproducibility fields that were previously missing:
  - git commit hash (best-effort; None outside a git checkout)
  - active model family / slug
  - baseline sampling (model-specific, per Model Card)
  - continuation sampling (uniform, Recum Table 15)
  - HuggingFace model config revision when derivable from the loaded tokenizer

Kept as a tiny standalone module so both scripts/run_adaptive_dialogue.py and
scripts/run_perturbed.py can enrich their metadata without duplicating logic.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def git_commit_hash() -> str | None:
    """Return the current HEAD commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes; None on error."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def tokenizer_model_revision(tokenizer: Any) -> str | None:
    """Extract the loaded HF model revision (commit hash) when available.

    Returns None for tokenizers that don't expose revision info (e.g. Mistral's
    custom tokenizer). Useful to pin the exact snapshot used for a run.
    """
    if tokenizer is None:
        return None
    for attr in ("_commit_hash", "commit_hash"):
        value = getattr(tokenizer, attr, None)
        if value:
            return str(value)
    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    if isinstance(init_kwargs, dict):
        for key in ("revision", "_commit_hash"):
            value = init_kwargs.get(key)
            if value:
                return str(value)
    return None


def build_provenance_block(
    *,
    model_name: str,
    backend: Any = None,
    baseline_sampling: dict | None = None,
    continuation_sampling: dict | None = None,
) -> dict:
    """Assemble the provenance sub-dict merged into run_metadata.json.

    Called by both run_adaptive_dialogue.write_run_metadata_summary and
    run_perturbed.write_run_metadata so any run — legacy or cross-model —
    carries reproducibility fields consistently.
    """
    try:
        from model_family import resolve_model_family  # noqa: WPS433

        family = resolve_model_family(model_name)
        family_name = family.name
        family_slug = family.slug
        think_tags = {
            "open": family.think_tags.open,
            "close": family.think_tags.close,
            "trigger": family.think_tags.trigger,
            "keep_in_history": family.think_tags.keep_in_history,
        }
        vllm_flags = list(family.vllm.flags)
    except Exception:
        family_name = family_slug = None
        think_tags = None
        vllm_flags = None

    tokenizer = getattr(backend, "tokenizer", None) if backend is not None else None
    return {
        "git_commit": git_commit_hash(),
        "git_dirty": git_dirty(),
        "model_family": family_name,
        "model_slug": family_slug,
        "think_tags": think_tags,
        "vllm_serve_flags": vllm_flags,
        "hf_model_revision": tokenizer_model_revision(tokenizer),
        "baseline_sampling": dict(baseline_sampling) if baseline_sampling else None,
        "continuation_sampling": dict(continuation_sampling) if continuation_sampling else None,
    }
