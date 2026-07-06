#!/usr/bin/env python3
"""
Script 3: Perturbed CoT runs (stop + recovery variants).

Loads verified baselines (Script 1) and generated injections (Script 2), builds
multi-turn dialogs with midpoint injection as a collaborating-agent note, and
runs stop (prefilled domain answer format) and recovery (natural completion).

The Stop variant constrains the model to a rapid-response regime
(stop_max_new_tokens, default 100 tokens) with domain-specific prefill. The
Recovery variant allows unconstrained self-correction up to recovery_max_tokens
(default 8192). The separate Stop cap keeps the REI denominator
(tokens_recovery - tokens_stop) interpretable.

Output: data/post_injection_runs.jsonl (input to Script 4 REI analysis).

What is stored (and what is not)
--------------------------------
Script 3 builds the corrupted dialog from baseline_cots.jsonl (prefix to midpoint)
plus injections_generated.jsonl (Claude steps in Turn 3). That full chain is the
MODEL INPUT — it is not written as one field to post_injection_runs.jsonl.

Each output row is one DeepSeek GPU run AFTER that injection: primarily raw_output
(what the experiment model generated next), extracted_answer, correct, variant
(stop vs recovery), and run metadata. To inspect the pre-injection prefix or the
injected steps, use baseline_cots.jsonl and injections_generated.jsonl.

Output JSONL schema (one JSON object per line):
  task_id, model, domain, eval_type — task and experiment model
  error_type, k, length_level, seed — design cell (matches Script 2 key)
  variant — stop (answer only) or recovery (continue reasoning)
  extracted_answer, gold_answer, correct — graded outcome
  raw_output — full model text for this run
  tokens_generated — completion length
  finish_reason — backend stop signal: stop (EOS) | length (token cap)
  prompt_tokens — input prompt length (tokenizer count)
  insertion_step — where the baseline prefix ended (floor(n/2) of baseline steps)
  recovery_terminated_by — recovery only: natural | token_limit | extraction_failed

Usage:
  python scripts/run_perturbed.py --dry-run --domain math --limit 5
  python scripts/run_perturbed.py --scope pilot --domain math
  python scripts/run_perturbed.py --dry-run --domain math --limit 3 --batch-size 4

Before GPU runs (non-dry-run), Script 3 preflights injections_generated.jsonl against
the selected scope: missing rows or structurally invalid injection_steps abort the run.
insertion_step metadata mismatches warn by default; use --strict-insertion-step to abort.
Use --skip-injection-preflight only for deliberate partial/debug runs.

Script 4 note: use recovery records with recovery_terminated_by == "natural" only for REI.

Reproducibility: each (task x condition) is run with the scope's fixed seeds
(full run: 42-46). Seeds are passed through per prompt to SamplingParams (vLLM,
as a parallel List[SamplingParams]) / torch.manual_seed (transformers). Bit-exact
reproducibility is NOT guaranteed under vLLM batching due to scheduler
non-determinism; the persisted raw_output is the ground truth. Run metadata
(model, vLLM/torch/CUDA versions, GPU) is written to results/run_metadata.json.

GPU one-task verification (run FIRST before any pilot/full sweep):
  python scripts/run_perturbed.py --config configs/experiment.yaml \\
      --scope pilot_gpu --domain math --limit 1
  Print the RENDERED PROMPT (not just raw_output) to confirm the baseline prefix
  is visible as plain text in the assistant turn (no <think> tags):
    from src.llm_backend import build_perturbed_messages, messages_to_prompt
    # after loading the tokenizer:
    print(messages_to_prompt(tokenizer, messages))
  Verify:
  1. Prefill continuation — the stop-variant output should begin with the
     expected answer format (e.g. "\\boxed{...}"), not re-emit the prefill.
  2. Recovery trace quality — the recovery-variant should show coherent
     reasoning after the injection note.
  3. Baseline visibility — the baseline prefix appears as PLAIN TEXT in the
     assistant turn (no <think>...</think> tags). DeepSeek-R1's chat template
     opens its own <think> block on the generation turn via add_generation_prompt.
  4. Fallback: set trace_in_user_turn: true in configs/experiment.yaml if
     assistant-turn mode produces unexpected template behaviour.

GPU SETUP (DeployGPU A100, CUDA 12.8 driver):
  pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
  pip uninstall vllm -y && pip install vllm  # reinstall after torch cu128
  Verify: python -c "import torch; print(torch.cuda.is_available())"
  Then set backend: vllm and batch_size: 32 in configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer_judge import resolve_det_tolerance, resolve_eval_type  # noqa: E402
from evaluator import (  # noqa: E402
    classify_recovery_termination,
    extract_domain_answer,
    is_correct,
)
from baseline_index import index_verified_baselines  # noqa: E402
from experiment_scope import design_lists, get_scope, print_run_plan, select_tasks  # noqa: E402
from experiment_design import injection_cell_key as injection_key  # noqa: E402
from grid_validation import (  # noqa: E402
    compute_midpoint,
    index_injections,
    injection_insertion_step,
    preflight_injections,
    validate_injection_record,
)
from llm_backend import (  # noqa: E402
    BaseBackend,
    DryRunBackend,
    TransformersBackend,
    build_perturbed_messages,
    create_backend,
)

DEFAULT_CONFIG = ROOT / "configs/experiment.yaml"

# Cross-model portage: stop-prefill is now built from the active model family's
# think close-tag + a domain suffix (Gemma/Ministral need different close tags
# than DeepSeek's </think>). The suffixes themselves are unchanged.
#
# Kept as module-level names for backward compat: existing tests / other
# scripts still import STOP_PREFILL_NUMERIC_SCIENCE / STOP_PREFILL_LOGIC.
from model_family import (  # noqa: E402
    STOP_PREFILL_LOGIC,
    STOP_PREFILL_NUMERIC_SCIENCE,
    build_stop_prefill,
)

# Legacy dict for callers that used to look up prefills directly. Kept
# non-empty so nobody accidentally hard-codes </think> going forward.
STOP_PREFILL = {
    # Populated per call via build_stop_prefill(); kept as a sentinel.
    "math": None,
}


def select_stop_prefill(domain: str, eval_type: str | None) -> str:
    """Choose the stop-variant prefill for the CURRENT active model family.

    Delegates to model_family.build_stop_prefill(), which uses
    `get_active_model_family().think_tags.close` — so DeepSeek gets
    `</think>\\n\\nTherefore...`, Gemma gets `<|channel|>\\n\\nTherefore...`,
    and Ministral gets `[/THINK]\\n\\nTherefore...`. Set the active family via
    `set_active_model_family(model_name)` before dialogue generation (done
    automatically inside `create_backend`).
    """
    return build_stop_prefill(domain, eval_type)


def deterministic_grade(
    *,
    extracted_answer: str | None,
    gold_answer: str,
    domain: str,
    eval_type: str | None,
    math_tolerance: float = 1e-4,
    science_tolerance: float = 1e-2,
) -> bool:
    """Deterministic-only correctness (no LLM judge).

    L0: the judge is decoupled from the GPU run. Script 3 writes det_correct only;
    scripts/judge_runs.py (Script 3.5) adds the LLM verdict post-hoc. This mirrors
    the det cross-check inside answer_judge.grade_sample without any API call.
    """
    resolved_eval_type = resolve_eval_type(domain, eval_type, gold_answer)
    det_tol = resolve_det_tolerance(
        domain,
        resolved_eval_type,
        math_tolerance=math_tolerance,
        science_tolerance=science_tolerance,
    )
    det_result = is_correct(
        extracted_answer,
        str(gold_answer),
        resolved_eval_type,
        math_tolerance=det_tol,
    )
    return bool(det_result) if det_result is not None else False


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_tasks(path: Path, domain: str | None) -> list[dict]:
    tasks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            if domain is None or task.get("domain") == domain:
                tasks.append(task)
    return tasks


def index_excluded_baselines(
    records: list[dict],
    model_name: str,
) -> dict[str, dict]:
    """task_id -> latest unverified baseline (for skip logging)."""
    out: dict[str, dict] = {}
    for rec in records:
        if rec.get("verified"):
            continue
        if rec.get("model") != model_name:
            continue
        out[rec["task_id"]] = rec
    return out


def report_injection_preflight(
    missing: list[str],
    invalid: list[str],
    insertion_mismatches: list[str],
    *,
    strict_insertion: bool,
) -> bool:
    """
    Print preflight results. Returns True if the run should abort (production only).
    """
    ok = True
    if missing:
        ok = False
        print(f"[PREFLIGHT FAIL] Missing {len(missing)} injection(s) for this scope:", file=sys.stderr)
        for key in missing[:25]:
            print(f"  - {key}", file=sys.stderr)
        if len(missing) > 25:
            print(f"  ... and {len(missing) - 25} more", file=sys.stderr)
        print("Run: python scripts/generate_injections.py", file=sys.stderr)

    if invalid:
        ok = False
        print(f"[PREFLIGHT FAIL] {len(invalid)} injection(s) structurally invalid:", file=sys.stderr)
        for msg in invalid[:25]:
            print(f"  - {msg}", file=sys.stderr)
        if len(invalid) > 25:
            print(f"  ... and {len(invalid) - 25} more", file=sys.stderr)
        print("Re-run Script 2 for the affected cells (or delete bad rows and resume).", file=sys.stderr)

    if insertion_mismatches:
        print(
            f"[PREFLIGHT WARN] {len(insertion_mismatches)} injection(s) have insertion_step "
            "metadata that differs from the current verified baseline midpoint.",
            file=sys.stderr,
        )
        for msg in insertion_mismatches[:10]:
            print(f"  - {msg}", file=sys.stderr)
        if len(insertion_mismatches) > 10:
            print(f"  ... and {len(insertion_mismatches) - 10} more", file=sys.stderr)
        print(
            "Script 3 uses the live baseline midpoint for trace construction. "
            "If baselines changed after Script 2, re-run generate_injections.py.",
            file=sys.stderr,
        )
        if strict_insertion:
            ok = False

    return not ok


def mock_injection_steps(k: int, error_type: str) -> list[str]:
    return [
        f"Step (injected {i + 1}): [{error_type}] Collaborating agent suggests a misleading step."
        for i in range(k)
    ]


def resolve_injection_steps(
    injection_index: dict[str, dict],
    task_id: str,
    error_type: str,
    k: int,
    length_level: str,
    *,
    dry_run: bool,
    allow_step1_only: bool = False,
) -> list[str]:
    key = injection_key(task_id, error_type, k, length_level)
    rec = injection_index.get(key)
    if rec:
        steps = list(rec.get("injection_steps") or [])
        issues = validate_injection_record(rec, k, allow_step1_only=allow_step1_only)
        if issues:
            raise KeyError(f"Invalid injection for {key}: {'; '.join(issues)}")
        return steps
    if dry_run:
        return mock_injection_steps(k, error_type)
    raise KeyError(f"Missing injection for {key} (run Script 2 or use --dry-run)")


def build_output_record(
    *,
    task: dict,
    baseline: dict,
    model_name: str,
    error_type: str,
    k: int,
    length_level: str,
    seed: int,
    variant: str,
    raw_output: str,
    insertion_step: int,
    tokens_generated: int,
    eval_type: str,
    finish_reason: str | None = None,
    prompt_tokens: int | None = None,
    recovery_terminated_by: str | None = None,
    stop_max_tokens: int | None = None,
    recovery_max_tokens: int | None = None,
    math_tolerance: float = 1e-4,
    science_tolerance: float = 1e-2,
) -> dict:
    domain = task.get("domain", baseline.get("domain", "math"))
    extracted = extract_domain_answer(
        raw_output, domain, eval_type, prefer_first_boxed=(variant == "stop")
    )
    gold = str(task.get("gold_answer", baseline.get("gold_answer", "")))
    # L0: deterministic grading only — the LLM judge runs post-hoc in
    # scripts/judge_runs.py, so the GPU run has no Anthropic dependency. correct
    # mirrors det_correct until judge_runs.py overwrites it with the verdict.
    det_correct = deterministic_grade(
        extracted_answer=extracted,
        gold_answer=gold,
        domain=domain,
        eval_type=eval_type,
        math_tolerance=math_tolerance,
        science_tolerance=science_tolerance,
    )

    record: dict = {
        "task_id": task["task_id"],
        "model": model_name,
        "error_type": error_type,
        "k": k,
        "length_level": length_level,
        "seed": seed,
        "variant": variant,
        "extracted_answer": extracted,
        "gold_answer": gold,
        "correct": det_correct,
        "verifier_score": int(det_correct),
        "judge_output": "",
        "judge_raw_output": "",
        "det_correct": det_correct,
        "judge_fallback": True,
        "tokens_generated": tokens_generated,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "raw_output": raw_output,
        "insertion_step": insertion_step,
        "domain": domain,
        "eval_type": eval_type,
    }
    # Per-row token caps so Script 4 classifies truncation per row (old rows
    # default to 100/8192 in evaluate.py).
    if stop_max_tokens is not None:
        record["stop_max_new_tokens"] = int(stop_max_tokens)
    if recovery_max_tokens is not None:
        record["recovery_max_tokens"] = int(recovery_max_tokens)
    if variant == "recovery":
        record["recovery_terminated_by"] = recovery_terminated_by
    return record


def load_existing_keys(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    keys: set[str] = set()
    for rec in load_jsonl(out_path):
        keys.add(
            f"{rec['task_id']}|{rec.get('model')}|{rec.get('error_type')}|"
            f"{rec.get('k')}|{rec.get('length_level')}|{rec.get('seed')}|{rec.get('variant')}"
        )
    return keys


def _default_batch_size(backend: BaseBackend, cfg_batch_size: int) -> int:
    """Return 1 for backends that don't benefit from batching, else cfg_batch_size."""
    if isinstance(backend, (DryRunBackend, TransformersBackend)):
        return 1
    return cfg_batch_size


def flush_batch(
    backend: BaseBackend,
    batch: list[dict],
    *,
    batch_index: int,
    total_batches: int | None,
    model_cfg: dict,
    out_path: Path,
    existing_keys: set[str],
    written_counter: list[int],
    math_tolerance: float = 1e-4,
    science_tolerance: float = 1e-2,
) -> None:
    """
    Process one collected batch: call generate_chat_batch(), write records.

    batch items are dicts with keys: messages, prefill, meta, variant, domain.

    This is the canonical inference path — all stop and recovery variants are
    dispatched through here (via maybe_flush in main).  Prefill selection uses
    select_stop_prefill(domain, eval_type) in the main loop before items are
    added to the batch; do NOT duplicate that logic here.
    """
    if not batch:
        return

    total_str = f"/{total_batches}" if total_batches is not None else ""
    print(f"[Batch {batch_index}{total_str}] flushing {len(batch)} prompts...")

    temperature = float(model_cfg.get("temperature", 0.6))
    top_p = float(model_cfg.get("top_p", 0.95))

    # Stop and recovery use different token budgets: Stop is a rapid-response
    # regime (short cap, prefilled answer format), recovery is unconstrained
    # self-correction. See stop_max_new_tokens / recovery_max_tokens in config.
    stop_indices = [i for i, item in enumerate(batch) if item["variant"] == "stop"]
    recovery_indices = [i for i, item in enumerate(batch) if item["variant"] == "recovery"]
    recovery_max_tokens = int(model_cfg.get("recovery_max_tokens", 8192))
    stop_max_tokens = int(model_cfg.get("stop_max_new_tokens", 100))

    results: list = [None] * len(batch)

    t0 = time.time()
    if stop_indices:
        stop_items = [(batch[i]["messages"], batch[i]["prefill"]) for i in stop_indices]
        stop_seeds = [batch[i]["meta"].get("seed") for i in stop_indices]
        stop_results = backend.generate_chat_batch(
            stop_items,
            max_new_tokens=stop_max_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=stop_seeds,
        )
        for idx, res in zip(stop_indices, stop_results):
            results[idx] = res

    if recovery_indices:
        rec_items = [(batch[i]["messages"], None) for i in recovery_indices]
        rec_seeds = [batch[i]["meta"].get("seed") for i in recovery_indices]
        rec_results = backend.generate_chat_batch(
            rec_items,
            max_new_tokens=recovery_max_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=rec_seeds,
        )
        for idx, res in zip(recovery_indices, rec_results):
            results[idx] = res

    elapsed = time.time() - t0
    total_tokens = sum(r.completion_tokens for r in results if r is not None)
    tok_s = total_tokens / elapsed if elapsed > 0 else 0.0
    print(f"[Batch {batch_index}{total_str}] done in {elapsed:.1f}s | {total_tokens} tokens | {tok_s:.0f} tok/s")

    for item, result in zip(batch, results):
        if result is None:
            continue
        meta = item["meta"]
        variant = item["variant"]
        domain = item["domain"]

        tokens = result.completion_tokens
        raw_output = result.text
        finish = result.finish_reason
        prompt_tokens = result.prompt_tokens

        if variant == "recovery":
            extracted_probe = extract_domain_answer(raw_output, domain, meta.get("eval_type"))
            recovery_terminated_by = classify_recovery_termination(
                finish_reason=finish,
                extracted_answer=extracted_probe,
                max_new_tokens=recovery_max_tokens,
                completion_tokens=tokens,
            )
        else:
            recovery_terminated_by = None

        record = build_output_record(
            task=meta["task"],
            baseline=meta["baseline"],
            model_name=meta["model_name"],
            error_type=meta["error_type"],
            k=meta["k"],
            length_level=meta["length_level"],
            seed=meta["seed"],
            variant=variant,
            raw_output=raw_output,
            insertion_step=meta["insertion_step"],
            tokens_generated=tokens,
            eval_type=meta["eval_type"],
            finish_reason=finish,
            prompt_tokens=prompt_tokens,
            recovery_terminated_by=recovery_terminated_by,
            stop_max_tokens=stop_max_tokens,
            recovery_max_tokens=recovery_max_tokens,
            math_tolerance=math_tolerance,
            science_tolerance=science_tolerance,
        )
        append_jsonl(out_path, record)
        existing_keys.add(meta["run_key"])
        written_counter[0] += 1

        status = (
            "OK"
            if record["correct"]
            else "WRONG"
            if record["correct"] is False
            else "NO_PARSE"
        )
        extra = f" | term={recovery_terminated_by}" if variant == "recovery" else ""
        print(
            f"  [{status}] {meta['task']['task_id']} | {variant:8} | {meta['error_type']:20} | "
            f"k={meta['k']} {meta['length_level']:5} | ins={meta['insertion_step']} | tok={tokens}{extra}"
        )


def write_run_metadata(
    out_dir: Path,
    *,
    scope_name: str,
    seeds: list,
    model_cfg: dict,
    recovery_max_tokens: int,
    stop_max_tokens: int,
    dry_run: bool,
    trace_in_user_turn: bool,
    reasoning_model: bool,
    allow_single_turn: bool,
    backend,
) -> Path:
    """Write results/run_metadata.json for reproducibility.

    Captures the scope/seed plan, sampling hyperparameters, and the runtime
    environment (model, vLLM/torch/CUDA versions, GPU). Bit-exact reproduction
    is not guaranteed under vLLM batching; raw_outputs in the run file are the
    ground truth, and this file documents the conditions that produced them.
    """
    import datetime
    import platform

    def _safe(fn):
        try:
            return fn()
        except Exception:
            return None

    vllm_version = _safe(lambda: __import__("vllm").__version__)
    torch_mod = _safe(lambda: __import__("torch"))
    torch_version = getattr(torch_mod, "__version__", None) if torch_mod else None
    cuda_version = getattr(getattr(torch_mod, "version", None), "cuda", None) if torch_mod else None
    gpu_name = None
    if torch_mod is not None:
        gpu_name = _safe(
            lambda: torch_mod.cuda.get_device_name(0) if torch_mod.cuda.is_available() else None
        )

    metadata = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script": "run_perturbed.py",
        "active_scope": scope_name,
        "seeds": list(seeds),
        "model": model_cfg.get("name"),
        "backend": model_cfg.get("backend"),
        "temperature": model_cfg.get("temperature"),
        "top_p": model_cfg.get("top_p"),
        "max_new_tokens": model_cfg.get("max_new_tokens"),
        "recovery_max_tokens": recovery_max_tokens,
        "stop_max_new_tokens": stop_max_tokens,
        "dry_run": dry_run,
        "trace_in_user_turn": trace_in_user_turn,
        "dialog_mode": "single_user_turn" if trace_in_user_turn else "three_turn",
        "reasoning_model": reasoning_model,
        "allow_single_turn_override": allow_single_turn,
        "prompt_render_method": getattr(backend, "prompt_render_method", "manual"),
        "vllm_version": vllm_version,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "python_version": platform.python_version(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run perturbed CoT traces (Script 3).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scope", type=str, default=None, help="Override active_scope (pilot/full).")
    parser.add_argument("--domain", type=str, default=None, help="Filter tasks/domain (e.g. math).")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks (quick test).")
    parser.add_argument("--dry-run", action="store_true", help="Mock backend; synthesize missing injections.")
    parser.add_argument(
        "--allow-single-turn",
        action="store_true",
        help="Override hard error when trace_in_user_turn=true on production runs (debug only).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prompts per vLLM call (default: 32 for vLLM, 1 for transformers/dry-run).",
    )
    parser.add_argument(
        "--skip-injection-preflight",
        action="store_true",
        help="Skip pre-GPU check for missing/invalid injections (debug partial runs only).",
    )
    parser.add_argument(
        "--strict-insertion-step",
        action="store_true",
        help="Abort when Script 2 insertion_step metadata differs from the live baseline midpoint.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    script3 = cfg.get("script3", {})
    paths = cfg.get("paths", {})
    model_cfg = dict(cfg["model"])
    model_name = model_cfg["name"]
    experiment_seed = int(cfg.get("experiment", {}).get("seed", 42))
    cfg_batch_size = int(script3.get("batch_size", 32))

    baseline_path = ROOT / script3.get("baseline_file", cfg.get("baseline", {}).get("generated_file", "data/baseline_cots.jsonl"))
    injections_path = ROOT / script3.get(
        "injections_file",
        cfg.get("injection", {}).get("generated_file", paths.get("injections", "data/injections_generated.jsonl")),
    )
    out_path = ROOT / script3.get("output_file", "data/post_injection_runs.jsonl")
    tasks_path = ROOT / paths.get("tasks", "data/experiment_tasks.jsonl")
    recovery_max_tokens = int(script3.get("recovery_max_tokens", 8192))
    stop_max_tokens = int(script3.get("stop_max_new_tokens", 100))
    script4_cfg = cfg.get("script4", {})
    math_tolerance = float(script4_cfg.get("math_tolerance", 1e-4))
    science_tolerance = float(script4_cfg.get("science_tolerance", 1e-2))
    reasoning_model = bool(script3.get("reasoning_model", model_cfg.get("reasoning_model", False)))
    trace_in_user_turn = bool(script3.get("trace_in_user_turn", False))
    if trace_in_user_turn and not args.dry_run and not args.allow_single_turn:
        raise ValueError(
            "trace_in_user_turn=true is a debug fallback only; refusing production runs "
            "with single-user-turn dialog. Use --allow-single-turn to override explicitly."
        )
    variants = list(script3.get("variants", ["stop", "recovery"]))

    scope = get_scope(cfg, args.scope)
    if args.domain:
        scope = dict(scope)
        scope["domains"] = [args.domain]
    # Per-scope task pool override (e.g. pilot_gpu points at the MATH-500/SciBench
    # file) so global paths.tasks / data/experiment_tasks.jsonl stays untouched.
    if scope.get("tasks"):
        tasks_path = ROOT / scope["tasks"]
    lists = design_lists(cfg, scope)
    perturbed_variants = [v for v in variants if v in ("stop", "recovery")]

    if not baseline_path.exists():
        print(f"Missing baselines: {baseline_path}", file=sys.stderr)
        print("Run: python scripts/generate_baselines.py", file=sys.stderr)
        sys.exit(1)

    if not injections_path.exists() and not args.dry_run:
        print(f"Missing injections: {injections_path}", file=sys.stderr)
        print("Run: python scripts/generate_injections.py", file=sys.stderr)
        sys.exit(1)

    baseline_records = load_jsonl(baseline_path)
    baseline_index = index_verified_baselines(baseline_records, model_name)
    excluded_index = index_excluded_baselines(baseline_records, model_name)
    injection_index = index_injections(load_jsonl(injections_path), target_model=model_name)

    all_tasks = load_tasks(tasks_path, args.domain)
    task_by_id = {t["task_id"]: t for t in all_tasks}
    eligible = [
        task_by_id[tid]
        for tid in sorted(baseline_index.keys())
        if tid in task_by_id
    ]
    tasks = select_tasks(eligible, scope, experiment_seed)
    if args.limit:
        tasks = tasks[: args.limit]

    if not args.dry_run and not args.skip_injection_preflight:
        missing, invalid, insertion_mismatches = preflight_injections(
            tasks, lists, injection_index, baseline_index
        )
        if report_injection_preflight(
            missing,
            invalid,
            insertion_mismatches,
            strict_insertion=args.strict_insertion_step,
        ):
            sys.exit(1)
    elif not args.dry_run and args.skip_injection_preflight:
        print("[PREFLIGHT] Skipped (--skip-injection-preflight). Missing cells will [SKIP] at runtime.")

    existing_keys = load_existing_keys(out_path)
    backend = create_backend(model_cfg, dry_run=args.dry_run)

    # Resolve effective batch size: CLI > config > auto-default based on backend type
    if args.batch_size is not None:
        effective_batch_size = args.batch_size
    else:
        effective_batch_size = _default_batch_size(backend, cfg_batch_size)

    print_run_plan(scope, tasks, lists, str(out_path))
    print(f"Baselines (verified, {model_name}): {len(baseline_index)}")
    print(f"Injections loaded: {len(injection_index)}" + (" (dry-run may synthesize)" if args.dry_run else ""))
    print(
        f"Variants: {perturbed_variants} | recovery_max_tokens={recovery_max_tokens} | "
        f"stop_max_new_tokens={stop_max_tokens}"
    )
    print(f"Backend: {'dry-run' if args.dry_run else model_cfg.get('backend', 'vllm')} | batch_size={effective_batch_size}")
    print(f"Output: {out_path}")
    print("")

    written = 0
    written_counter = [0]  # mutable int for flush_batch
    skipped = 0
    missing_baseline = 0

    # Collect batch items; flush when batch reaches effective_batch_size
    pending_batch: list[dict] = []
    batch_index = 1
    insertion_mismatch_warnings: set[str] = set()

    def maybe_flush(force: bool = False) -> None:
        nonlocal batch_index
        if not pending_batch:
            return
        if force or len(pending_batch) >= effective_batch_size:
            flush_batch(
                backend,
                pending_batch,
                batch_index=batch_index,
                total_batches=None,
                model_cfg={
                    **model_cfg,
                    "recovery_max_tokens": recovery_max_tokens,
                    "stop_max_new_tokens": stop_max_tokens,
                },
                out_path=out_path,
                existing_keys=existing_keys,
                written_counter=written_counter,
                math_tolerance=math_tolerance,
                science_tolerance=science_tolerance,
            )
            pending_batch.clear()
            batch_index += 1

    for task in tasks:
        baseline = baseline_index.get(task["task_id"])
        if baseline is None:
            missing_baseline += 1
            excluded = excluded_index.get(task["task_id"])
            if excluded:
                reason = excluded.get("exclude_reason", "unverified")
                steps = excluded.get("step_count", "?")
                print(
                    f"[SKIP] {task['task_id']} — baseline excluded "
                    f"(reason={reason}, steps={steps})"
                )
            else:
                print(f"[SKIP] {task['task_id']} — no baseline for model")
            continue

        steps = list(baseline.get("steps") or [])
        if not steps:
            missing_baseline += 1
            print(f"[SKIP] {task['task_id']} — baseline has no steps")
            continue

        insertion_step, baseline_prefix = compute_midpoint(steps)
        eval_type = task.get("eval_type", "exact")
        problem = task["prompt"]
        domain = task.get("domain", "math")

        if args.dry_run and hasattr(backend, "gold_hint"):
            backend.gold_hint = str(task.get("gold_answer", ""))

        for run_seed in lists["seeds"]:
            for error_type in lists["error_types"]:
                for k in lists["k_injections"]:
                    for length_level in lists["length_levels"]:
                        try:
                            inj_key = injection_key(
                                task["task_id"], error_type, k, length_level
                            )
                            injection_rec = injection_index.get(inj_key)
                            injection_steps = resolve_injection_steps(
                                injection_index,
                                task["task_id"],
                                error_type,
                                k,
                                length_level,
                                dry_run=args.dry_run,
                            )
                        except KeyError as exc:
                            print(f"[SKIP] {task['task_id']} | {error_type} k={k} {length_level}: {exc}")
                            continue

                        recorded_insertion_step = injection_insertion_step(injection_rec)
                        if (
                            recorded_insertion_step is not None
                            and recorded_insertion_step != insertion_step
                            and inj_key not in insertion_mismatch_warnings
                        ):
                            insertion_mismatch_warnings.add(inj_key)
                            print(
                                f"[WARN] {inj_key}: injection metadata insertion_step="
                                f"{recorded_insertion_step}, current baseline insertion_step={insertion_step}; "
                                "using current baseline insertion_step for trace construction."
                            )

                        messages = build_perturbed_messages(
                            problem,
                            baseline_prefix,
                            injection_steps,
                            reasoning_model=reasoning_model,
                            trace_in_user_turn=trace_in_user_turn,
                        )

                        for variant in perturbed_variants:
                            run_key = (
                                f"{task['task_id']}|{model_name}|{error_type}|{k}|"
                                f"{length_level}|{run_seed}|{variant}"
                            )
                            if run_key in existing_keys:
                                skipped += 1
                                continue

                            prefill = select_stop_prefill(domain, eval_type) if variant == "stop" else None

                            pending_batch.append({
                                "messages": messages,
                                "prefill": prefill,
                                "variant": variant,
                                "domain": domain,
                                "meta": {
                                    "task": task,
                                    "baseline": baseline,
                                    "model_name": model_name,
                                    "error_type": error_type,
                                    "k": k,
                                    "length_level": length_level,
                                    "seed": run_seed,
                                    "insertion_step": insertion_step,
                                    "eval_type": eval_type,
                                    "run_key": run_key,
                                },
                            })
                            maybe_flush()

    # Flush any remaining items
    maybe_flush(force=True)

    written = written_counter[0]
    print(f"\nDone. Written: {written} | Skipped (existing): {skipped} | Missing baseline: {missing_baseline}")
    if args.dry_run:
        print("Dry-run complete (no GPU). Validate chat/prefill on one real task before full pilot.")
    print(f"Output: {out_path}")

    meta_path = write_run_metadata(
        ROOT / "results",
        scope_name=scope.get("name", args.scope or "unknown"),
        seeds=lists["seeds"],
        model_cfg=model_cfg,
        recovery_max_tokens=recovery_max_tokens,
        stop_max_tokens=stop_max_tokens,
        dry_run=args.dry_run,
        trace_in_user_turn=trace_in_user_turn,
        reasoning_model=reasoning_model,
        allow_single_turn=args.allow_single_turn,
        backend=backend,
    )
    print(f"Run metadata: {meta_path}")


if __name__ == "__main__":
    main()
