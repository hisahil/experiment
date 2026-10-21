"""Per-model-family configuration for cross-model experiments.

Loads `configs/model_families.yaml` and resolves a `ModelFamilyConfig` from
any HuggingFace slug. Used by:
  - src/llm_backend.py           (ThinkTags in build_continuation_prompt)
  - src/evaluator.py             (_strip_thinking_suffix close-tag)
  - src/adaptive_dialogue.py     (strip_think_closure via active family)
  - scripts/generate_baselines.py (baseline_sampling per Model Card)
  - scripts/run_adaptive_dialogue.py (stop-prefill + continuation sampling)
  - scripts/run_perturbed.py     (stop-prefill)
  - scripts/smoke_cross_model.py (vLLM launch flags)

Cross-model sampling rules (Recum Table 15, verified in configs):
  1. Baseline (Script 1)       — model-specific (Model Card values here)
  2. Injection (Script 2)      — uniform Claude Sonnet 4.6 (unrelated to family)
  3. Continuation (Script 3b)  — uniform for all families (continuation_sampling)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "model_families.yaml"

# Domain-specific stop prefills; suffix follows the model's think close-tag.
STOP_PREFILL_MATH_SUFFIX = "\n\nTherefore, the final answer is \\boxed{"
STOP_PREFILL_NUMERIC_SCIENCE = "\n\nTherefore, the final answer is "
STOP_PREFILL_LOGIC = "\n\nFinal answer: "


@dataclass(frozen=True)
class ThinkTags:
    """Reasoning delimiters for one model family."""
    open: str = "<think>"
    close: str = "</think>"
    trigger: str | None = None
    keep_in_history: bool = False


@dataclass(frozen=True)
class VLLMConfig:
    """vLLM serve/engine parameters for one family."""
    flags: tuple[str, ...] = field(default_factory=tuple)
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.92
    tokenizer_mode: str | None = None
    config_format: str | None = None
    load_format: str | None = None
    stop_tokens: tuple[str, ...] = field(default_factory=tuple)
    limit_mm_per_prompt: dict[str, int] | None = None
    hf_overrides: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelFamilyConfig:
    """Everything needed to run one model family through the pipeline."""
    name: str
    slug: str
    think_tags: ThinkTags
    baseline_sampling: dict[str, float | int]
    tokenizer_extra_kwargs: dict[str, Any]
    vllm: VLLMConfig
    continuation_sampling: dict[str, float | int] = field(default_factory=dict)
    baseline_prompt: str | None = None
    baseline_use_chat_template: bool = False
    # Whether Script 1 should strip everything up to and including the model's
    # closing think tag from raw_output before segmentation. Decoupled from
    # baseline_use_chat_template so families that need the strip (GLM writes
    # "Step N:" INSIDE the think block) can keep the flat-text prompt aligned
    # with DeepSeek. See scripts/generate_baselines.py.
    baseline_strip_think: bool = False


DEFAULT_THINK_TAGS = ThinkTags()


def _load_raw_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_think_tags(raw: dict[str, Any] | None) -> ThinkTags:
    if not raw:
        return DEFAULT_THINK_TAGS
    return ThinkTags(
        open=str(raw.get("open", DEFAULT_THINK_TAGS.open)),
        close=str(raw.get("close", DEFAULT_THINK_TAGS.close)),
        trigger=raw.get("trigger"),
        keep_in_history=bool(raw.get("keep_in_history", False)),
    )


def _parse_vllm(raw: dict[str, Any] | None) -> VLLMConfig:
    if not raw:
        return VLLMConfig()
    serve = raw.get("vllm_serve") or {}
    engine = raw.get("vllm_engine") or {}
    stop_tokens_raw = raw.get("stop_tokens") or []
    return VLLMConfig(
        flags=tuple(str(f) for f in (serve.get("flags") or [])),
        max_model_len=int(engine.get("max_model_len", 16384)),
        gpu_memory_utilization=float(engine.get("gpu_memory_utilization", 0.92)),
        tokenizer_mode=(
            str(engine["tokenizer_mode"]) if engine.get("tokenizer_mode") else None
        ),
        config_format=(
            str(engine["config_format"]) if engine.get("config_format") else None
        ),
        load_format=(
            str(engine["load_format"]) if engine.get("load_format") else None
        ),
        stop_tokens=tuple(str(t) for t in stop_tokens_raw),
        limit_mm_per_prompt=(
            {str(k): int(v) for k, v in engine["limit_mm_per_prompt"].items()}
            if engine.get("limit_mm_per_prompt")
            else None
        ),
        hf_overrides=(
            dict(engine["hf_overrides"]) if engine.get("hf_overrides") else None
        ),
    )


def _family_entry(name: str, entry: dict[str, Any], continuation: dict) -> ModelFamilyConfig:
    return ModelFamilyConfig(
        name=name,
        slug=str(entry.get("slug", name)),
        think_tags=_parse_think_tags(entry.get("think_tags")),
        baseline_sampling=dict(entry.get("baseline_sampling") or {}),
        tokenizer_extra_kwargs=dict(entry.get("tokenizer_extra_kwargs") or {}),
        vllm=_parse_vllm(entry),
        continuation_sampling=dict(continuation or {}),
        baseline_prompt=entry.get("baseline_prompt"),
        baseline_use_chat_template=bool(entry.get("baseline_use_chat_template", False)),
        baseline_strip_think=bool(entry.get("baseline_strip_think", False)),
    )


@lru_cache(maxsize=4)
def _cached_families(
    config_path: str,
) -> tuple[
    tuple[tuple[tuple[str, ...], ModelFamilyConfig], ...],
    ModelFamilyConfig,
]:
    """Ordered (patterns, config) pairs + default. Cached per absolute path."""
    raw = _load_raw_config(Path(config_path))
    continuation = dict(raw.get("continuation_sampling") or {})
    default_entry = raw.get("default") or {}
    default_cfg = _family_entry("default", default_entry, continuation)

    ordered: list[tuple[tuple[str, ...], ModelFamilyConfig]] = []
    for name, entry in (raw.get("families") or {}).items():
        patterns = tuple(str(p) for p in (entry.get("match") or []))
        ordered.append((patterns, _family_entry(str(name), entry, continuation)))
    return tuple(ordered), default_cfg


def resolve_model_family(
    model_name: str,
    *,
    config_path: Path | None = None,
) -> ModelFamilyConfig:
    """Match `model_name` against configured substrings; fall back to default.

    Matching is case-insensitive and substring-based (first hit wins). This lets
    us match both `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` and
    `DeepSeek-R1-Distill-Qwen-32B` with the same pattern, and covers subclassed
    slugs like `Ministral-3-14B-Reasoning-2512` via the `Ministral-3-14B` prefix.
    """
    path = str((config_path or DEFAULT_CONFIG).resolve())
    ordered, default_cfg = _cached_families(path)
    lowered = model_name.lower()
    for patterns, cfg in ordered:
        if any(p.lower() in lowered for p in patterns):
            return cfg
    return default_cfg


# Process-wide active family. Set by scripts once the target model is known so
# helpers that cannot easily pass the family through (e.g. evaluator answer
# parsing) can query it cheaply.
_active_family: ModelFamilyConfig | None = None


def set_active_model_family(
    model_name: str, *, config_path: Path | None = None
) -> ModelFamilyConfig:
    """Set process-wide active family and return the resolved config."""
    global _active_family
    _active_family = resolve_model_family(model_name, config_path=config_path)
    return _active_family


def get_active_model_family() -> ModelFamilyConfig:
    """Return the active family (falls back to DeepSeek default when unset)."""
    if _active_family is None:
        return resolve_model_family("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    return _active_family


def get_think_tags(model_name: str | None = None) -> ThinkTags:
    """Convenience: think tags for `model_name` or the active family."""
    if model_name:
        return resolve_model_family(model_name).think_tags
    return get_active_model_family().think_tags


def apply_baseline_sampling(model_cfg: dict, model_name: str) -> dict:
    """Overlay Script-1 baseline sampling from the model family onto model_cfg.

    Returns a new dict; original is not mutated. Keys not set by the family
    (e.g. max_new_tokens) survive from `model_cfg`. Explicit Card values from
    the family win over experiment.yaml defaults.
    """
    family = resolve_model_family(model_name)
    merged = dict(model_cfg)
    for key, value in family.baseline_sampling.items():
        merged[key] = value
    return merged


def continuation_sampling_kwargs(model_name: str | None = None) -> dict:
    """Recum-konforme uniform Continuation sampling (temp=0.6, top_p=0.95)."""
    family = resolve_model_family(model_name) if model_name else get_active_model_family()
    return dict(family.continuation_sampling)


def build_stop_prefill(
    domain: str,
    eval_type: str | None,
    *,
    think_close_tag: str | None = None,
) -> str:
    """Build stop-variant prefill: close think block, then domain-specific suffix.

    - math: `<close_tag><math_suffix>` (default: closes think + `\\boxed{`)
    - science + numeric: numeric-suffix WITHOUT think close (matches existing
      run_perturbed.py behaviour where non-math domains use plain suffix)
    - logic: logic-suffix WITHOUT think close
    """
    close = think_close_tag or get_think_tags().close
    if domain == "science" and eval_type == "numeric":
        return STOP_PREFILL_NUMERIC_SCIENCE
    if domain == "logic":
        return STOP_PREFILL_LOGIC
    # math (and fallback)
    return f"{close}{STOP_PREFILL_MATH_SUFFIX}"


def make_stop_prefill_selector(
    model_name: str,
) -> Callable[[str, str | None], str]:
    """Return a select_stop_prefill(domain, eval_type) closure for one model."""
    tags = resolve_model_family(model_name).think_tags

    def select(domain: str, eval_type: str | None) -> str:
        return build_stop_prefill(domain, eval_type, think_close_tag=tags.close)

    return select


def strip_thinking_suffix(text: str, model_name: str | None = None) -> str:
    """Prefer model text after a closing thinking tag when present.

    Model-agnostic version of evaluator._strip_thinking_suffix — uses the
    resolved family's close-tag. Falls back to `</think>` for the default
    family, matching legacy behaviour.
    """
    close = get_think_tags(model_name).close
    idx = text.rfind(close)
    if idx == -1:
        return text
    after = text[idx + len(close):].strip()
    return after or text


def vllm_launch_argv(model_name: str) -> list[str]:
    """Build the CLI arg list for `vllm serve <model> ...` for this family."""
    family = resolve_model_family(model_name)
    args = ["vllm", "serve", model_name]
    args.extend(family.vllm.flags)
    args.extend(["--max-model-len", str(family.vllm.max_model_len)])
    args.extend(["--gpu-memory-utilization", str(family.vllm.gpu_memory_utilization)])
    return args


def vllm_python_api_kwargs(model_name: str) -> dict[str, Any]:
    """Family-specific kwargs for `vllm.LLM(...)` Python-API construction.

    Mistral/Ministral requires `tokenizer_mode="mistral"`,
    `config_format="mistral"`, `load_format="mistral"` — otherwise the HF
    tokenizer route is used, which does not know the Mistral chat template
    and the model emits its own `[INST]`/`[/INST]`/`[SYSTEM_PROMPT]` tokens
    into the output (chat-template leak).

    These fields are separate from `vllm_serve.flags` because `flags` targets
    the `vllm serve` CLI, while our `VLLMBackend` uses the Python API which
    accepts kwargs, not CLI args.
    """
    family = resolve_model_family(model_name)
    kwargs: dict[str, Any] = {}
    if family.vllm.tokenizer_mode:
        kwargs["tokenizer_mode"] = family.vllm.tokenizer_mode
    if family.vllm.config_format:
        kwargs["config_format"] = family.vllm.config_format
    if family.vllm.load_format:
        kwargs["load_format"] = family.vllm.load_format
    if family.vllm.limit_mm_per_prompt is not None:
        kwargs["limit_mm_per_prompt"] = dict(family.vllm.limit_mm_per_prompt)
    if family.vllm.hf_overrides is not None:
        kwargs["hf_overrides"] = dict(family.vllm.hf_overrides)
    return kwargs


def family_stop_tokens(model_name: str | None = None) -> list[str]:
    """Family-specific extra stop tokens for `SamplingParams(stop=[...])`.

    Currently used for Ministral so the model does not continue generating
    after emitting its own `[/INST]` / `[INST]` tokens (self-simulated new
    turns). Empty for families without such artefacts (DeepSeek, GLM, Gemma).
    """
    family = resolve_model_family(model_name) if model_name else get_active_model_family()
    return list(family.vllm.stop_tokens)


def model_slug(model_name: str) -> str:
    """Filesystem-safe slug for `data/baselines/<slug>/`, `results/<slug>/`.

    Uses the family's explicit `slug:` field when the family matches (avoids
    the `Ministral-3-14B` → `14b` collision with DeepSeek-14B). Falls back to
    the tail of the HF slug when nothing matches, preserving legacy
    per-checkpoint slugs like `1.5b`, `7b`, `14b`, `32b` for DeepSeek.
    """
    family = resolve_model_family(model_name)
    if family.name != "default":
        # DeepSeek is special: keep the historical per-checkpoint slugs.
        if family.name == "deepseek":
            return _deepseek_checkpoint_slug(model_name)
        return family.slug
    return _fallback_short_slug(model_name)


_DEEPSEEK_SLUGS = {
    "1.5b": ["1.5B", "1.5b"],
    "7b": ["7B", "7b"],
    "14b": ["14B", "14b"],
    "32b": ["32B", "32b"],
}


def _deepseek_checkpoint_slug(model_name: str) -> str:
    for slug, patterns in _DEEPSEEK_SLUGS.items():
        if any(p in model_name for p in patterns):
            return slug
    return "deepseek"


def _fallback_short_slug(model_name: str) -> str:
    tail = model_name.rsplit("/", 1)[-1].lower()
    for token in tail.replace("-", " ").split():
        if token.endswith("b") and token[:-1].replace(".", "").isdigit():
            return token
    return tail
