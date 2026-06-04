#!/usr/bin/env python3
"""
Script 3: Perturbed CoT runs (stop + recovery variants).

Loads verified baselines (Script 1) and generated injections (Script 2), builds
multi-turn dialogs with midpoint injection as a collaborating-agent note, and
runs stop (prefilled domain answer format) and recovery (natural completion).

Output: data/perturbed_runs.jsonl (input to Script 4 REI analysis).

Usage:
  python scripts/run_perturbed.py --dry-run --domain math --limit 5
  python scripts/run_perturbed.py --scope pilot --domain math
  python scripts/run_perturbed.py --dry-run --domain math --limit 3 --batch-size 4

Script 4 note: use recovery records with recovery_terminated_by == "natural" only for REI.

# GPU SETUP (DeployGPU A100, CUDA 12.8 driver):
# pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
# pip uninstall vllm -y && pip install vllm  # reinstall after torch cu128
# Verify: python -c "import torch; print(torch.cuda.is_available())"
# Then set backend: vllm and batch_size: 32 in configs/experiment.yaml
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

from evaluator import (  # noqa: E402
    classify_recovery_termination,
    extract_domain_answer,
    is_correct,
)
from experiment_scope import design_lists, get_scope, print_run_plan, select_tasks  # noqa: E402
from llm_backend import (  # noqa: E402
    BaseBackend,
    DryRunBackend,
    TransformersBackend,
    build_perturbed_messages,
    create_backend,
)

DEFAULT_CONFIG = ROOT / "configs/experiment.yaml"

STOP_PREFILL = {
    "math": "\n\nTherefore, the final answer is \\boxed{",
    "science": "\n\nTherefore, the correct answer is option ",
    "code": "\n\nHere is the final implementation:\n```python\n",
}

# Open-ended numeric science (SciBench) and logic (K&K) need different stop-variant
# prefills than the MMLU MCQ / math-boxed defaults above.
STOP_PREFILL_NUMERIC_SCIENCE = "\n\nTherefore, the final answer is "
STOP_PREFILL_LOGIC = "\n\nFinal answer: "


def select_stop_prefill(domain: str, eval_type: str | None) -> str:
    """Choose the stop-variant prefill by domain and eval_type."""
    if domain == "science" and eval_type == "numeric":
        return STOP_PREFILL_NUMERIC_SCIENCE
    if domain == "logic":
        return STOP_PREFILL_LOGIC
    return STOP_PREFILL.get(domain, STOP_PREFILL["math"])


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


def index_verified_baselines(
    records: list[dict],
    model_name: str,
) -> dict[str, dict]:
    """task_id -> baseline record (verified only, matching model)."""
    out: dict[str, dict] = {}
    for rec in records:
        if not rec.get("verified"):
            continue
        if rec.get("model") != model_name:
            continue
        out[rec["task_id"]] = rec
    return out


def index_injections(records: list[dict]) -> dict[str, dict]:
    """task_id|error_type|k|length_level -> injection record."""
    out: dict[str, dict] = {}
    for rec in records:
        key = f"{rec['task_id']}|{rec['error_type']}|{rec['k']}|{rec['length_level']}"
        out[key] = rec
    return out


def compute_midpoint(steps: list[str]) -> tuple[int, str]:
    midpoint = len(steps) // 2
    prefix = "\n".join(steps[:midpoint])
    return midpoint, prefix


def injection_insertion_step(rec: dict | None) -> int | None:
    """Return Script 2 insertion metadata when present; legacy records omit it."""
    if not rec:
        return None
    value = rec.get("insertion_step", rec.get("midpoint_step"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
) -> list[str]:
    key = f"{task_id}|{error_type}|{k}|{length_level}"
    rec = injection_index.get(key)
    if rec:
        return list(rec.get("injection_steps") or [])
    if dry_run:
        return mock_injection_steps(k, error_type)
    raise KeyError(f"Missing injection for {key} (run Script 2 or use --dry-run)")


def run_stop_variant(
    backend,
    messages: list[dict],
    domain: str,
    model_cfg: dict,
) -> tuple[str, int, str | None]:
    prefill = STOP_PREFILL.get(domain, STOP_PREFILL["math"])
    max_new = int(model_cfg.get("max_new_tokens", 2048))
    result = backend.generate_chat(
        messages,
        max_new_tokens=max_new,
        temperature=float(model_cfg.get("temperature", 0.6)),
        top_p=float(model_cfg.get("top_p", 0.95)),
        prefill=prefill,
    )
    return result.text, result.completion_tokens, result.finish_reason


def run_recovery_variant(
    backend,
    messages: list[dict],
    model_cfg: dict,
    recovery_max_tokens: int,
) -> tuple[str, int, str | None]:
    result = backend.generate_chat(
        messages,
        max_new_tokens=recovery_max_tokens,
        temperature=float(model_cfg.get("temperature", 0.6)),
        top_p=float(model_cfg.get("top_p", 0.95)),
        prefill=None,
    )
    return result.text, result.completion_tokens, result.finish_reason


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
    midpoint_step: int,
    tokens_generated: int,
    eval_type: str,
    recovery_terminated_by: str | None = None,
) -> dict:
    domain = task.get("domain", baseline.get("domain", "math"))
    extracted = extract_domain_answer(raw_output, domain, eval_type)
    gold = str(task.get("gold_answer", baseline.get("gold_answer", "")))
    correct = is_correct(extracted, gold, eval_type)

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
        "correct": correct,
        "tokens_generated": tokens_generated,
        "raw_output": raw_output,
        "midpoint_step": midpoint_step,
        "insertion_step": midpoint_step,
        "domain": domain,
    }
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
) -> None:
    """
    Process one collected batch: call generate_chat_batch(), write records.

    batch items are dicts with keys: messages, prefill, meta, variant, domain.
    """
    if not batch:
        return

    total_str = f"/{total_batches}" if total_batches is not None else ""
    print(f"[Batch {batch_index}{total_str}] flushing {len(batch)} prompts...")

    max_new = int(model_cfg.get("max_new_tokens", 2048))
    temperature = float(model_cfg.get("temperature", 0.6))
    top_p = float(model_cfg.get("top_p", 0.95))

    # Split stop vs recovery (different max_new_tokens for recovery)
    stop_indices = [i for i, item in enumerate(batch) if item["variant"] == "stop"]
    recovery_indices = [i for i, item in enumerate(batch) if item["variant"] == "recovery"]
    recovery_max_tokens = int(model_cfg.get("recovery_max_tokens", 8192))

    results: list = [None] * len(batch)

    t0 = time.time()
    if stop_indices:
        stop_items = [(batch[i]["messages"], batch[i]["prefill"]) for i in stop_indices]
        stop_results = backend.generate_chat_batch(
            stop_items,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
        )
        for idx, res in zip(stop_indices, stop_results):
            results[idx] = res

    if recovery_indices:
        rec_items = [(batch[i]["messages"], None) for i in recovery_indices]
        rec_results = backend.generate_chat_batch(
            rec_items,
            max_new_tokens=recovery_max_tokens,
            temperature=temperature,
            top_p=top_p,
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
            midpoint_step=meta["midpoint_step"],
            tokens_generated=tokens,
            eval_type=meta["eval_type"],
            recovery_terminated_by=recovery_terminated_by,
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
            f"k={meta['k']} {meta['length_level']:5} | mid={meta['midpoint_step']} | tok={tokens}{extra}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run perturbed CoT traces (Script 3).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scope", type=str, default=None, help="Override active_scope (pilot/full).")
    parser.add_argument("--domain", type=str, default=None, help="Filter tasks/domain (e.g. math).")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks (pilot smoke).")
    parser.add_argument("--dry-run", action="store_true", help="Mock backend; synthesize missing injections.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prompts per vLLM call (default: 32 for vLLM, 1 for transformers/dry-run).",
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
    out_path = ROOT / script3.get("output_file", "data/perturbed_runs.jsonl")
    tasks_path = ROOT / paths.get("tasks", "data/experiment_tasks.jsonl")
    recovery_max_tokens = int(script3.get("recovery_max_tokens", 8192))
    reasoning_model = bool(script3.get("reasoning_model", model_cfg.get("reasoning_model", False)))
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

    baseline_index = index_verified_baselines(load_jsonl(baseline_path), model_name)
    injection_index = index_injections(load_jsonl(injections_path))

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
    print(f"Variants: {perturbed_variants} | recovery_max_tokens={recovery_max_tokens}")
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
                model_cfg={**model_cfg, "recovery_max_tokens": recovery_max_tokens},
                out_path=out_path,
                existing_keys=existing_keys,
                written_counter=written_counter,
            )
            pending_batch.clear()
            batch_index += 1

    for task in tasks:
        baseline = baseline_index.get(task["task_id"])
        if baseline is None:
            missing_baseline += 1
            print(f"[SKIP] {task['task_id']} — no verified baseline for model")
            continue

        steps = list(baseline.get("steps") or [])
        if not steps:
            missing_baseline += 1
            print(f"[SKIP] {task['task_id']} — baseline has no steps")
            continue

        midpoint_step, baseline_prefix = compute_midpoint(steps)
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
                            injection_key = f"{task['task_id']}|{error_type}|{k}|{length_level}"
                            injection_rec = injection_index.get(injection_key)
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
                            and recorded_insertion_step != midpoint_step
                            and injection_key not in insertion_mismatch_warnings
                        ):
                            insertion_mismatch_warnings.add(injection_key)
                            print(
                                f"[WARN] {injection_key}: injection metadata insertion_step="
                                f"{recorded_insertion_step}, current baseline midpoint={midpoint_step}; "
                                "using current baseline midpoint for trace construction."
                            )

                        messages = build_perturbed_messages(
                            problem,
                            baseline_prefix,
                            injection_steps,
                            reasoning_model=reasoning_model,
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
                                    "midpoint_step": midpoint_step,
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


if __name__ == "__main__":
    main()
