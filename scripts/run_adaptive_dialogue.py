#!/usr/bin/env python3
"""
Script 3b: Adaptive alternating dialogue runs (stop + recovery).

Orchestrates multi-turn dialogs with real intermediate GPU generations and
adaptive Claude injections (injection 1 pre-generated, injections 2..k adaptive).

Usage:
  python scripts/run_adaptive_dialogue.py --dry-run --domain math --limit 2
  python scripts/run_adaptive_dialogue.py --scope pilot_gpu_adaptive --domain math
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluator import (  # noqa: E402
    classify_recovery_termination,
    extract_domain_answer,
)
from adaptive_dialogue import (  # noqa: E402
    DialogueState,
    InjectionBreaker,
    InjectionCircuitBreakerError,
    append_jsonl_flush,
    append_transcript_jsonl,
    build_complete_dialogue_record,
    compute_continuation_prefix,
    init_dialogue_state,
    resolve_intermediate_max_tokens,
    run_continuation_dialogues,
    run_dialogue_until_complete,
)
from experiment_design import (  # noqa: E402
    GENERATION_MODE_CONTINUATION,
    GENERATION_MODE_RESTART,
    GENERATION_MODES,
)
from baseline_index import index_verified_baselines  # noqa: E402
from experiment_scope import (  # noqa: E402
    apply_task_shard,
    count_planned_dialogues,
    design_lists,
    get_scope,
    parse_task_shard,
    print_run_plan,
    select_tasks,
    select_tasks_balanced,
)
from llm_backend import MATH_RECOVERY_FORMAT_INSTRUCTION, create_backend  # noqa: E402
from model_family import (  # noqa: E402
    continuation_sampling_kwargs,
    make_stop_prefill_selector,
    resolve_model_family,
    set_active_model_family,
)
from run_timings import build_extrapolation, fmt_duration, write_run_timings  # noqa: E402
from run_perturbed import (  # noqa: E402
    compute_midpoint,
    deterministic_grade,
    index_injections,
    injection_key,
    load_jsonl,
    load_tasks,
    load_yaml,
    resolve_injection_steps,
)

DEFAULT_CONFIG = ROOT / "configs/experiment.yaml"


def build_injection_provider(
    *,
    dry_run: bool,
    baseline_contexts: dict[str, dict],
    adaptive_from: int,
    claude_fn=None,
    max_tokens_by_level: dict[str, int] | None = None,
):
    from generate_injections import (  # noqa: E402
        generate_adaptive_step,
        placeholder_adaptive_step,
    )

    # Per-length_level budgets for live adaptive steps (short=400, long=1200).
    level_tokens = dict(max_tokens_by_level or {"short": 400, "long": 1200})
    default_tokens = level_tokens.get("short", 400)

    def get_injection(state: DialogueState) -> str:
        step_index = state.current_round()
        step_idx0 = step_index - 1
        neutral = state.condition == "neutral_control"
        ctx = baseline_contexts.get(state.task["task_id"])

        if (
            step_index == 1
            and state.pregenerated_first
            and state.condition == "adversarial"
            and not dry_run
        ):
            return state.pregenerated_first

        if dry_run:
            return placeholder_adaptive_step(
                state.task,
                state.error_type,
                state.k,
                step_idx0,
                list(state.prior_injections),
                list(state.model_turns),
                neutral=neutral,
            )

        if claude_fn is None:
            raise RuntimeError("Claude provider required for non-dry-run adaptive injections")

        # Live steps 2..k must use the per-length_level budget (long needs 1200,
        # not the hardcoded short=400 that truncated long injections in the pilot).
        step_max_tokens = level_tokens.get(state.length_level, default_tokens)

        def _gen(system: str, user: str) -> str:
            return claude_fn(system, user, step_max_tokens)

        step, _role = generate_adaptive_step(
            state.task,
            state.error_type,
            state.k,
            state.length_level,
            step_idx0,
            list(state.prior_injections),
            list(state.model_turns),
            ctx,
            _gen,
            neutral=neutral,
        )
        if step_index >= adaptive_from or neutral:
            state.agent_role_id = state.agent_role_id or (
                "persistent_neutral_control" if neutral else f"persistent_{state.error_type}"
            )
        return step

    return get_injection


def make_claude_fn(provider: str, model: str, temperature: float):
    """Build the live-injection Claude caller.

    The returned function takes max_tokens per call so adaptive steps can pick the
    budget by length_level (short=400, long=1200) instead of a single global cap.
    """
    from generate_injections import call_llm, parse_single_step  # noqa: E402

    def claude_fn(system: str, user: str, max_tokens: int) -> str:
        raw, _stop_reason = call_llm(
            user,
            provider,
            model,
            max_tokens,
            temperature,
            system=system,
        )
        return parse_single_step(raw)

    return claude_fn


def build_output_record(
    *,
    state: DialogueState,
    variant: str,
    raw_output: str,
    tokens_generated: int,
    finish_reason: str | None,
    prompt_tokens: int | None,
    recovery_max_tokens: int,
    stop_max_tokens: int,
    math_tolerance: float = 1e-4,
    science_tolerance: float = 1e-2,
    gen: object | None = None,
) -> dict:
    task = state.task
    domain = task.get("domain", "math")
    eval_type = task.get("eval_type")
    from first_commit import apply_empty_output, extract_stop_commitment

    extracted = extract_domain_answer(
        raw_output, domain, eval_type, prefer_first_boxed=(variant == "stop")
    )
    gold = str(task.get("gold_answer", ""))
    # L0: deterministic grading only — LLM judge runs post-hoc (scripts/judge_runs.py).
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
        "model": state.model_name,
        "error_type": state.error_type if state.condition == "adversarial" else None,
        "k": state.k,
        "length_level": state.length_level,
        "seed": state.seed,
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
        "insertion_step": state.insertion_step,
        "domain": domain,
        "eval_type": eval_type,
        "dialogue_id": state.dialogue_id,
        "n_rounds": state.k,
        "condition": state.condition,
        "turn2_source": state.turn2_source,
        "dialogue_mode": (
            "continuation" if state.generation_mode == "continuation" else "adaptive_alternating"
        ),
        "generation_mode": state.generation_mode,
        # Per-row token caps so Script 4 classifies truncation per row (old rows
        # default to 100/8192). Lets the 100-cap and 500-cap runs be reported
        # stratified instead of mixed.
        "stop_max_new_tokens": int(stop_max_tokens),
        "recovery_max_tokens": int(recovery_max_tokens),
    }
    if variant == "stop":
        commitment = extract_stop_commitment(
            raw_output,
            domain=domain,
            eval_type=eval_type,
            model_name=state.model_name,
        )
        record["first_commit_answer"] = commitment["answer"]
        record["first_commit_span"] = commitment["span"]
        record["first_commit_incomplete"] = commitment["incomplete"]
    if apply_empty_output(record):
        record["det_correct"] = False
    if variant == "recovery":
        record["recovery_terminated_by"] = classify_recovery_termination(
            finish_reason=finish_reason,
            extracted_answer=extracted,
            max_new_tokens=recovery_max_tokens,
            completion_tokens=tokens_generated,
        )

    # Phase-1 metadata (Script 4 / derived_metrics inputs). decoy_match is NOT
    # computed here — derived_metrics owns the single is_correct() decoy path so
    # the matching rule lives in one place (Phase 2). We only persist the decoy
    # target + baseline step count.
    if state.decoy_answer is not None:
        record["decoy_answer"] = state.decoy_answer
    if state.baseline_step_count is not None:
        record["baseline_step_count"] = state.baseline_step_count
    # Injection sub-labels (Punkt 3, Feld-Propagation). Only meaningful for the
    # matching error_type; None otherwise. Downstream evaluation stratifies by
    # these to reveal e.g. paraphrased_shift subtype effects on susceptibility.
    if state.shift_subtype is not None:
        record["shift_subtype"] = state.shift_subtype
    if state.off_topic_theme is not None:
        record["off_topic_theme"] = state.off_topic_theme

    # Phase-1 instrumentation off the GenerationResult (continuation only). Missing
    # on dry-run/restart → fields simply stay absent.
    if gen is not None:
        for attr in (
            "prompt_sha256",
            "reasoning_trace_tokens",
            "mean_logprob",
            "batch_wall_time_s",
            "batch_size",
            "approx_wall_time_s",
            "tokens_per_second_batch",
            "tokens_per_second_sample_approx",
        ):
            val = getattr(gen, attr, None)
            if val is not None:
                record[attr] = val
        # Boxed-answer distribution only on stop (decoy-vs-gold in Phase 2).
        if variant == "stop":
            top_lp = getattr(gen, "answer_top_logprobs", None)
            if top_lp is not None:
                record["answer_top_logprobs"] = top_lp
    return record


def _write_prompt_audit(audit_dir: Path, state, stop_res, rec_res) -> None:
    """Persist the full rendered prompt for one dialogue (continuation proof).

    Writes data/prompt_audit/{dialogue_id}/{stop,recovery}.txt with the exact
    string sent to the backend, plus meta.json (hashes, token counts). Lets a
    reviewer confirm the think block is OPEN and the trace + injections are present
    (no restart). All rows still carry prompt_sha256 for reproducibility.
    """
    try:
        d = Path(audit_dir) / str(state.dialogue_id)
        d.mkdir(parents=True, exist_ok=True)
        meta: dict = {
            "dialogue_id": state.dialogue_id,
            "task_id": state.task.get("task_id"),
            "model": state.model_name,
            "error_type": state.error_type,
            "k": state.k,
            "length_level": state.length_level,
            "seed": state.seed,
        }
        for variant, res in (("stop", stop_res), ("recovery", rec_res)):
            rendered = getattr(res, "rendered_prompt", None)
            if rendered is not None:
                (d / f"{variant}.txt").write_text(rendered, encoding="utf-8")
            meta[f"{variant}_prompt_sha256"] = getattr(res, "prompt_sha256", None)
            meta[f"{variant}_reasoning_trace_tokens"] = getattr(
                res, "reasoning_trace_tokens", None
            )
            meta[f"{variant}_prompt_tokens"] = getattr(res, "prompt_tokens", None)
        (d / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        print(f"[WARN] prompt audit write failed for {state.dialogue_id}: {exc}", file=sys.stderr)


def load_existing_keys(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    keys: set[str] = set()
    for rec in load_jsonl(out_path):
        cond = rec.get("condition", "adversarial")
        et = rec.get("error_type") or "neutral"
        keys.add(
            f"{rec['task_id']}|{rec.get('model')}|{et}|{rec.get('k')}|"
            f"{rec.get('length_level')}|{rec.get('seed')}|{rec.get('variant')}|{cond}"
        )
    return keys


def check_grid_completeness(
    out_path: Path,
    tasks: list[dict],
    lists: dict,
    model_name: str,
) -> list[tuple]:
    """Compare expected adversarial stop/recovery grid cells against written rows.

    Uses the (balanced) selected `tasks` — the same selection the run used — so the
    expectation matches the real run, not the full pool. Returns the sorted list of
    missing run-cell keys (empty when complete) and prints a short warning.
    """
    from experiment_design import DesignAxes, expected_run_keys, run_cell_key

    axes = DesignAxes(
        variants=list(lists.get("variants") or ["stop", "recovery"]),
        error_types=list(lists.get("error_types") or []),
        k_injections=[int(k) for k in lists.get("k_injections") or [1]],
        length_levels=list(lists.get("length_levels") or ["short"]),
        seeds=[int(s) for s in lists.get("seeds") or [42]],
    )
    expected = expected_run_keys(
        tasks, axes, model=model_name, variants=["stop", "recovery"]
    )
    actual: set[tuple] = set()
    for rec in load_jsonl(out_path):
        if rec.get("condition", "adversarial") == "neutral_control":
            continue
        if rec.get("model") != model_name:
            continue
        if rec.get("variant") not in ("stop", "recovery"):
            continue
        try:
            actual.add(
                run_cell_key(
                    rec["task_id"],
                    rec.get("error_type"),
                    int(rec["k"]),
                    rec["length_level"],
                    int(rec["seed"]),
                    rec["variant"],
                    model_name,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    missing = sorted(expected - actual, key=lambda t: tuple(str(x) for x in t))
    if missing:
        print(
            f"\n[GRID WARNING] {len(missing)} of {len(expected)} adversarial grid "
            f"cell(s) missing (balanced {len(tasks)}-task expectation):"
        )
        for cell in missing[:20]:
            print(f"  - {'|'.join(str(x) for x in cell)}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    else:
        print(
            f"\n[GRID OK] All {len(expected)} adversarial grid cells present "
            f"(balanced {len(tasks)}-task expectation)."
        )
    return missing


def append_jsonl(path: Path, record: dict, *, flush: bool = False) -> None:
    if flush:
        append_jsonl_flush(path, record)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_key_for_record(model_name: str, out: dict) -> str:
    et = out.get("error_type") or "neutral"
    return (
        f"{out['task_id']}|{model_name}|{et}|{out['k']}|"
        f"{out['length_level']}|{out['seed']}|{out['variant']}|{out['condition']}"
    )


def write_run_metadata_summary(
    *,
    output_dir: Path,
    scope_name: str | None,
    model_name: str,
    wall_time_s: float,
    dialogues_completed: int,
    stopped_reason: str,
    backend: object | None = None,
    continuation_sampling: dict | None = None,
    is_continuation: bool | None = None,
) -> None:
    meta_path = output_dir / "run_metadata.json"
    payload: dict = {}
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}

    # Cross-model provenance: git-Commit + HF-Revision + family + sampling.
    # Keeps metadata compatible with old callers (nothing removed).
    try:
        from run_provenance import build_provenance_block  # noqa: WPS433

        provenance = build_provenance_block(
            model_name=model_name,
            backend=backend,
            continuation_sampling=continuation_sampling,
        )
    except Exception:
        provenance = {}

    dialog_mode = (
        "continuation" if is_continuation
        else ("adaptive_alternating" if is_continuation is False else "adaptive_alternating")
    )
    payload.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": "run_adaptive_dialogue.py",
        "active_scope": scope_name,
        "model": model_name,
        "wall_time_s": round(wall_time_s, 1),
        "dialogues_completed": dialogues_completed,
        "stopped_reason": stopped_reason,
        "dialogue_mode": dialog_mode,
    })
    payload.update({k: v for k, v in provenance.items() if v is not None})
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def should_run_neutral(scope: dict, script3: dict, cfg: dict) -> bool:
    if scope.get("include_neutral_control"):
        return True
    subset = script3.get("neutral_control_subset") or {}
    if subset.get("enabled"):
        return True
    return False


def neutral_seeds(scope: dict, script3: dict, lists: dict) -> list:
    subset = script3.get("neutral_control_subset") or {}
    if subset.get("enabled") and subset.get("seeds"):
        return list(subset["seeds"])
    if scope.get("neutral_seeds"):
        return list(scope["neutral_seeds"])
    return list(lists["seeds"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive alternating dialogue runs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scope", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--balanced-per-domain", type=int, default=None)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--record-timings", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--prompt-audit-sample",
        type=int,
        default=None,
        help=(
            "Override script3.prompt_audit_sample: persist the FULL rendered prompt "
            "for the first N dialogues (continuation-mode proof). All rows still carry "
            "prompt_sha256. Use a large N on mini_run to audit every dialogue."
        ),
    )
    parser.add_argument(
        "--generation-mode",
        choices=GENERATION_MODES,
        default=None,
        help=(
            "Override script3.dialogue_generation_mode. continuation = Recum-style "
            "open think block, deletion model, all k injections pre-generated (no live "
            "Claude). restart = legacy multi-turn dialogue."
        ),
    )
    parser.add_argument(
        "--task-shard",
        type=str,
        default=None,
        help=(
            "Data-parallel sharding 'I/N' (0-based I, N total). Each shard runs the "
            "disjoint subset tasks[I::N] so N GPUs/pods split the run with no overlap. "
            "Combine with --out so shards write separate files; merge after the run."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Override script3.output_file. Use a per-shard path (e.g. "
            "data/shards/post_injection_runs_shard0.jsonl) when running --task-shard "
            "across nodes to avoid concurrent-append conflicts on shared storage."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Override model.name from configs/experiment.yaml. Required when "
            "swapping cross-model families (glm-z1-9b / gemma-4-12b / "
            "ministral-3-14b) without a per-model overlay config."
        ),
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help=(
            "Override script3.continuation.baseline_file — path to the "
            "per-model continuation baselines JSONL "
            "(data/baselines/<slug>/baseline_cots_continuation.jsonl)."
        ),
    )
    parser.add_argument(
        "--injections-file",
        type=Path,
        default=None,
        help=(
            "Override script3.injections_file / paths.injections — path to the "
            "per-model injections JSONL (data/injections_generated_<slug>.jsonl)."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help=(
            "Override paths.tasks — path to the task pool JSONL "
            "(e.g. data/smoke_tasks.jsonl for a smoke run vs "
            "data/experiment_tasks.jsonl for the full 100-task battery)."
        ),
    )
    parser.add_argument(
        "--complete-dialogues-file",
        type=Path,
        default=None,
        help=(
            "Override script3.complete_dialogues_file — full-dialogue transcript "
            "path (data/<slug>/complete_dialogues.jsonl)."
        ),
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help=(
            "Deprecated no-op: the LLM judge is always decoupled now (L0). Rows are "
            "written with deterministic grading only (judge_fallback=true); re-grade "
            "post-hoc via scripts/judge_runs.py."
        ),
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    script3 = cfg.get("script3", {})
    paths = cfg.get("paths", {})
    model_cfg = dict(cfg["model"])
    # CLI-Override wins (cross-model portage): --model-name lets the smoke
    # script and per-model orchestrators pick a family without a bespoke
    # overlay config.
    if args.model_name:
        model_cfg["name"] = args.model_name
    model_name = model_cfg["name"]
    experiment_seed = int(cfg.get("experiment", {}).get("seed", 42))

    # Cross-model portage: resolve the family up-front so stop-prefill uses the
    # right close-tag (Gemma <|channel|>, Ministral [/THINK], DeepSeek/GLM
    # </think>) and continuation sampling stays uniform 0.6/0.95 (Recum).
    family = resolve_model_family(model_name)
    set_active_model_family(model_name)
    select_stop_prefill = make_stop_prefill_selector(model_name)
    cont_sampling = continuation_sampling_kwargs(model_name)
    if cont_sampling:
        # Uniform Recum continuation sampling wins over experiment.yaml
        # (DeepSeek runs happened to match already; Gemma/Ministral baselines
        # differ but continuation must be uniform for a fair comparison).
        for key, value in cont_sampling.items():
            model_cfg[key] = value
    print(
        f"[cross-model] family={family.name} slug={family.slug} "
        f"think_close={family.think_tags.close!r} "
        f"continuation={cont_sampling}"
    )

    # Generation mode: CLI override > config > restart default.
    generation_mode = (
        args.generation_mode
        or script3.get("dialogue_generation_mode", GENERATION_MODE_RESTART)
    )
    if generation_mode not in GENERATION_MODES:
        print(f"Unknown generation mode: {generation_mode!r}", file=sys.stderr)
        sys.exit(1)
    continuation_cfg = script3.get("continuation", {}) or {}
    is_continuation = generation_mode == GENERATION_MODE_CONTINUATION

    if script3.get("dialogue_mode", "three_turn") != "adaptive_alternating" and not args.dry_run:
        print(
            "Warning: script3.dialogue_mode is not adaptive_alternating; "
            "set it in configs/experiment.yaml for production adaptive runs.",
            file=sys.stderr,
        )

    # Continuation mode reads the \n\n-joined continuation baselines (Step N:
    # prefixes stripped); restart mode reads the legacy baseline file.
    # CLI --baseline-file wins over both continuation.baseline_file and
    # script3.baseline_file (cross-model orchestration).
    if args.baseline_file is not None:
        baseline_path = args.baseline_file if args.baseline_file.is_absolute() else ROOT / args.baseline_file
    elif is_continuation:
        baseline_path = ROOT / continuation_cfg.get(
            "baseline_file", script3.get("baseline_file", "data/baseline_cots.jsonl")
        )
    else:
        baseline_path = ROOT / script3.get("baseline_file", "data/baseline_cots.jsonl")
    if args.injections_file is not None:
        injections_path = (
            args.injections_file
            if args.injections_file.is_absolute()
            else ROOT / args.injections_file
        )
    else:
        injections_path = ROOT / script3.get(
            "injections_file",
            paths.get("injections", "data/injections_generated.jsonl"),
        )
    out_path = (
        args.out if args.out is not None
        else ROOT / script3.get("output_file", "data/post_injection_runs.jsonl")
    )
    transcript_path = ROOT / script3.get(
        "adaptive_dialogues_file", "data/adaptive_dialogues.jsonl"
    )
    if args.complete_dialogues_file is not None:
        complete_path = (
            args.complete_dialogues_file
            if args.complete_dialogues_file.is_absolute()
            else ROOT / args.complete_dialogues_file
        )
    else:
        complete_path = ROOT / script3.get(
            "complete_dialogues_file", "data/complete_dialogues.jsonl"
        )
    # adaptive_dialogues.jsonl is a redundant subset of complete_dialogues.jsonl in
    # continuation mode; default OFF. Set script3.write_adaptive_dialogues: true to
    # re-enable for the legacy restart path.
    write_adaptive_transcript = bool(script3.get("write_adaptive_dialogues", False))
    # Prompt audit: persist the FULL rendered prompt (continuation-mode proof) for
    # the first N dialogues; all rows still carry prompt_sha256 + token counts.
    prompt_audit_sample = (
        int(args.prompt_audit_sample)
        if args.prompt_audit_sample is not None
        else int(script3.get("prompt_audit_sample", 0))
    )
    prompt_audit_dir = ROOT / script3.get("prompt_audit_dir", "data/prompt_audit")
    # Logprobs (perplexity proxy + boxed-answer distribution). int = top-k per token,
    # None disables. Only the vLLM backend fills them; others ignore.
    _capture_lp = script3.get("capture_logprobs", None)
    capture_logprobs = (
        int(_capture_lp) if _capture_lp not in (None, False, "", 0, "0") else None
    )
    timings_path = ROOT / cfg.get("script4", {}).get("output_dir", "results") / "run_timings.json"
    # CLI --tasks wins over paths.tasks and scope.tasks (smoke needs to point
    # at data/smoke_tasks.jsonl without editing the base config).
    if args.tasks is not None:
        tasks_path = args.tasks if args.tasks.is_absolute() else ROOT / args.tasks
    else:
        tasks_path = ROOT / paths.get("tasks", "data/experiment_tasks.jsonl")
    recovery_max_tokens = int(script3.get("recovery_max_tokens", 8192))
    stop_max_tokens = int(script3.get("stop_max_new_tokens", 100))
    script4_cfg = cfg.get("script4", {})
    math_tolerance = float(script4_cfg.get("math_tolerance", 1e-4))
    science_tolerance = float(script4_cfg.get("science_tolerance", 1e-2))
    injection_tag = str(script3.get("injection_tag", "none"))
    turn2_source = str(script3.get("turn2_source", "baseline_prefix"))
    adaptive_from = int(script3.get("adaptive_from_injection", 2))
    intermediate_cfg = script3.get("intermediate_max_tokens", "auto")

    scope = get_scope(cfg, args.scope)
    if args.domain:
        scope = dict(scope)
        scope["domains"] = [args.domain]
    # CLI --tasks takes final precedence (cross-model smoke uses
    # data/smoke_tasks.jsonl). scope.tasks only overrides when no CLI arg.
    if scope.get("tasks") and args.tasks is None:
        tasks_path = ROOT / scope["tasks"]
    lists = design_lists(cfg, scope)

    baseline_records = load_jsonl(baseline_path)
    baseline_index = index_verified_baselines(baseline_records, model_name)
    _inj_records = load_jsonl(injections_path)
    injection_index = index_injections(_inj_records, target_model=model_name)
    if not injection_index and _inj_records:
        print(
            f"[injections] No rows tagged target_model={model_name!r}; "
            f"indexing all {len(_inj_records)} rows (cross-model reuse).",
            file=sys.stderr,
        )
        injection_index = index_injections(_inj_records, target_model=None)


    all_tasks = load_tasks(tasks_path, args.domain)
    task_by_id = {t["task_id"]: t for t in all_tasks}
    eligible = [task_by_id[tid] for tid in sorted(baseline_index.keys()) if tid in task_by_id]
    if args.balanced_per_domain is not None:
        tasks = select_tasks_balanced(
            eligible, scope, experiment_seed, args.balanced_per_domain,
        )
    else:
        tasks = select_tasks(eligible, scope, experiment_seed)
    if args.limit:
        tasks = tasks[: args.limit]

    # Data-parallel sharding: tasks are already sorted by select_tasks, so the
    # interleaved partition tasks[I::N] is deterministic, disjoint across shards,
    # and unions back to the full set. Applied last so --limit and shard compose.
    shard = parse_task_shard(args.task_shard)
    if shard is not None:
        n_before = len(tasks)
        tasks = apply_task_shard(tasks, shard)
        print(
            f"[SHARD] {shard[0]}/{shard[1]} — {len(tasks)}/{n_before} tasks on this shard"
        )

    intermediate_max_tokens = resolve_intermediate_max_tokens(
        intermediate_cfg,
        [baseline_index[t["task_id"]] for t in tasks if t["task_id"] in baseline_index],
    )

    from grid_validation import preflight_injections  # noqa: E402
    from run_perturbed import report_injection_preflight  # noqa: E402

    if not args.dry_run:
        # Adaptive dialogues only consume injection_steps[0]; Script 2 may pre-generate
        # a single round-1 step (injection.pregenerate_steps). Pass the config so the
        # preflight tolerates trimmed records (and still accepts legacy full-k sets).
        dialogue_mode = script3.get("dialogue_mode", "three_turn")
        pregenerate_steps = (cfg.get("injection") or {}).get("pregenerate_steps", "auto")
        missing, invalid, mismatches = preflight_injections(
            tasks, lists, injection_index, baseline_index,
            pregenerate_steps=pregenerate_steps,
            dialogue_mode=dialogue_mode,
        )
        if report_injection_preflight(missing, invalid, mismatches, strict_insertion=False):
            print("[PREFLIGHT] Aborting before GPU load — fix injections first.", file=sys.stderr)
            sys.exit(1)
        print(f"[PREFLIGHT] OK — {len(tasks)} tasks, injections complete")

    model_load_t0 = time.time()
    backend = create_backend(model_cfg, dry_run=args.dry_run)
    model_load_s = time.time() - model_load_t0

    existing_keys = load_existing_keys(out_path)
    run_neutral = should_run_neutral(scope, script3, cfg)
    neutral_seed_list = neutral_seeds(scope, script3, lists) if run_neutral else []
    planned_dialogues = count_planned_dialogues(tasks, lists, include_neutral=run_neutral)
    skipped_existing = 0

    from generate_injections import load_baseline_contexts, MAX_TOKENS  # noqa: E402
    baseline_contexts = load_baseline_contexts(baseline_path, model_name)

    inj_cfg = cfg.get("injection", {})
    claude_fn = None
    # Continuation mode uses only pre-generated injections (no live Claude).
    if not args.dry_run and not is_continuation:
        provider = inj_cfg.get("provider", "anthropic")
        claude_model = inj_cfg.get("model", "claude-sonnet-4-6")
        claude_fn = make_claude_fn(
            provider,
            claude_model,
            float(inj_cfg.get("temperature", 0.8)),
        )

    get_injection = build_injection_provider(
        dry_run=args.dry_run,
        baseline_contexts=baseline_contexts,
        adaptive_from=adaptive_from,
        claude_fn=claude_fn,
        max_tokens_by_level=MAX_TOKENS,
    )

    # Continuation mode caps intermediate turns at one paragraph (\n\n stop); the
    # token cap is a safety net only.
    trace_separator = str(continuation_cfg.get("trace_separator", "\n\n"))
    intermediate_stop = str(continuation_cfg.get("intermediate_stop", "\n\n"))
    if is_continuation:
        intermediate_max_tokens = int(
            continuation_cfg.get("intermediate_max_tokens", intermediate_max_tokens)
        )

    print_run_plan(scope, tasks, lists, str(out_path))
    print(
        f"Generation mode: {generation_mode} | "
        f"intermediate_max_tokens={intermediate_max_tokens}"
        + (f" | intermediate_stop={intermediate_stop!r}" if is_continuation else "")
    )
    print(f"Planned dialogues: {planned_dialogues} | Neutral control: {run_neutral}")
    print(f"Output: {out_path}")
    print(f"Complete dialogues: {complete_path}")
    if args.budget_hours is not None:
        print(f"Budget: {args.budget_hours}h compute (model_load_s excluded)")
    print(f"Model load: {model_load_s:.1f}s")
    print("")

    states: list[DialogueState] = []
    skipped = 0

    from convert_baselines_to_continuation import strip_step_prefix  # noqa: E402

    for task in tasks:
        baseline = baseline_index.get(task["task_id"])
        if not baseline:
            continue
        steps = list(baseline.get("steps") or [])
        if not steps:
            continue
        if is_continuation:
            cont_steps = list(baseline.get("continuation_steps") or [])
            if not cont_steps:
                # Fallback when pointed at a non-converted baseline: strip in memory.
                cont_steps = [s for s in (strip_step_prefix(x) for x in steps) if s]
            insertion_step, baseline_prefix = compute_continuation_prefix(cont_steps)
        else:
            insertion_step, baseline_prefix = compute_midpoint(steps)

        conditions: list[tuple[str, str | None]] = []
        for error_type in lists["error_types"]:
            conditions.append(("adversarial", error_type))
        if run_neutral:
            conditions.append(("neutral_control", None))

        for condition, error_type in conditions:
            seeds = (
                neutral_seed_list if condition == "neutral_control" else lists["seeds"]
            )
            for run_seed in seeds:
                for k in lists["k_injections"]:
                    for length_level in lists["length_levels"]:
                        et_key = error_type or "neutral"
                        run_key_stop = (
                            f"{task['task_id']}|{model_name}|{et_key}|{k}|"
                            f"{length_level}|{run_seed}|stop|{condition}"
                        )
                        run_key_rec = (
                            f"{task['task_id']}|{model_name}|{et_key}|{k}|"
                            f"{length_level}|{run_seed}|recovery|{condition}"
                        )
                        if run_key_stop in existing_keys and run_key_rec in existing_keys:
                            skipped += 2
                            skipped_existing += 1
                            continue

                        pregenerated_first: str | None = None
                        agent_role_id: str | None = None
                        all_inj_steps: list[str] = []
                        shift_subtype: str | None = None
                        off_topic_theme: str | None = None
                        if condition == "adversarial":
                            try:
                                inj_steps = resolve_injection_steps(
                                    injection_index,
                                    task["task_id"],
                                    error_type,
                                    k,
                                    length_level,
                                    dry_run=args.dry_run,
                                    # Continuation consumes ALL k pre-generated steps;
                                    # restart/adaptive only needs inj_steps[0].
                                    allow_step1_only=not is_continuation,
                                )
                                all_inj_steps = list(inj_steps)
                                pregenerated_first = inj_steps[0] if inj_steps else None
                                agent_role_id = f"persistent_{error_type}"
                                # Pull the injection sub-labels off the same
                                # index-lookup key used by resolve_injection_steps.
                                # Present only for the relevant error_type; None
                                # for the others (Punkt 3, Feld-Propagation).
                                inj_key = injection_key(
                                    task["task_id"], error_type, k, length_level
                                )
                                inj_rec = injection_index.get(inj_key) or {}
                                shift_subtype = inj_rec.get("shift_subtype")
                                off_topic_theme = inj_rec.get("off_topic_theme")
                            except KeyError as exc:
                                if not args.dry_run:
                                    print(f"[SKIP] {task['task_id']} {error_type} k={k}: {exc}")
                                    continue

                        state = init_dialogue_state(
                            task=task,
                            baseline=baseline,
                            baseline_prefix=baseline_prefix,
                            insertion_step=insertion_step,
                            error_type=error_type or "neutral",
                            k=int(k),
                            length_level=length_level,
                            seed=int(run_seed),
                            condition=condition,  # type: ignore[arg-type]
                            model_name=model_name,
                            turn2_source=turn2_source,
                            pregenerated_first=pregenerated_first,
                            agent_role_id=agent_role_id,
                            injection_tag=injection_tag,
                        )
                        if is_continuation:
                            state.generation_mode = GENERATION_MODE_CONTINUATION
                            state.accumulated_trace = baseline_prefix
                            if baseline_prefix:
                                state.add_trace_segment("baseline_prefix", baseline_prefix)
                            state.injection_steps = all_inj_steps
                            # baseline_step_count = clean continuation steps used for
                            # the midpoint split (Phase-2 recovery-inflation denominator).
                            state.baseline_step_count = len(cont_steps)
                            # Decoy only meaningful for false_consensus; null otherwise.
                            if error_type == "false_consensus":
                                state.decoy_answer = task.get("decoy_answer")
                        # Injection sub-labels (Punkt 3) — always propagate what
                        # Script 2 recorded; they're None for non-matching error
                        # types (e.g. shift_subtype only exists on paraphrased_shift).
                        state.shift_subtype = shift_subtype
                        state.off_topic_theme = off_topic_theme
                        states.append(state)

    if not states:
        print(f"No new dialogues to run. Skipped (existing): {skipped}")
        return

    compute_t0 = time.monotonic()
    budget_deadline = None
    if args.budget_hours is not None and args.budget_hours > 0:
        budget_deadline = compute_t0 + args.budget_hours * 3600.0

    written = 0
    dialogues_completed = 0
    audited = 0
    progress_start = time.time()

    def print_progress() -> None:
        if dialogues_completed <= 0:
            return
        elapsed = time.time() - progress_start
        avg = elapsed / dialogues_completed
        remaining = max(0, planned_dialogues - skipped_existing - dialogues_completed)
        eta = avg * remaining
        extrap = build_extrapolation(
            completed_dialogues=dialogues_completed,
            compute_elapsed_s=elapsed,
            planned_tonight=planned_dialogues,
            scope=scope.get("name"),
        )
        full_days = extrap.get("extrapolated_full_thesis_days")
        full_cost = extrap.get("extrapolated_full_cost_usd")
        extra = ""
        if full_days is not None and full_cost is not None:
            extra = f" | FULL THESIS ~{full_days}d ~${full_cost:.0f}"
        print(
            f"    progress: {dialogues_completed}/{planned_dialogues - skipped_existing} dialogues "
            f"| elapsed {fmt_duration(elapsed)} | avg {avg:.0f}s/dialogue "
            f"| ETA tonight {fmt_duration(eta)}{extra}"
        )

    def on_dialogue_complete(state, stop_res, rec_res) -> None:
        nonlocal written, dialogues_completed, audited
        for variant, res in (("stop", stop_res), ("recovery", rec_res)):
            out = build_output_record(
                state=state,
                variant=variant,
                raw_output=res.text,
                tokens_generated=res.completion_tokens,
                finish_reason=res.finish_reason,
                prompt_tokens=res.prompt_tokens,
                recovery_max_tokens=recovery_max_tokens,
                stop_max_tokens=stop_max_tokens,
                math_tolerance=math_tolerance,
                science_tolerance=science_tolerance,
                gen=res,
            )
            append_jsonl(out_path, out, flush=True)
            existing_keys.add(run_key_for_record(model_name, out))
            written += 1
        complete_rec = build_complete_dialogue_record(state, stop_res, rec_res)
        append_jsonl(complete_path, complete_rec, flush=True)
        # Prompt audit: persist the FULL rendered prompt for the first N dialogues
        # (continuation-mode proof: open think block + trace actually present).
        if audited < prompt_audit_sample:
            _write_prompt_audit(prompt_audit_dir, state, stop_res, rec_res)
            audited += 1
        # adaptive_dialogues.jsonl: redundant subset, off by default (see config).
        if write_adaptive_transcript:
            append_transcript_jsonl(transcript_path, state)
        dialogues_completed += 1
        print_progress()

    injection_breaker = InjectionBreaker(
        threshold=int(script3.get("injection_failure_threshold", 5))
    )
    try:
        if is_continuation:
            # Math recovery turns get a \boxed{} format instruction (parser-friendly);
            # only when every dialogue is math so non-math output is not constrained.
            math_only = all(
                st.task.get("domain", "math") == "math" for st in states
            )
            recovery_system = MATH_RECOVERY_FORMAT_INSTRUCTION if math_only else None
            # Deletion model: all k injections pre-generated, no live Claude barrier.
            records, completed, failed, stopped_reason = run_continuation_dialogues(
                states,
                backend,
                intermediate_max_tokens=intermediate_max_tokens,
                stop_max_tokens=stop_max_tokens,
                recovery_max_tokens=recovery_max_tokens,
                temperature=float(model_cfg.get("temperature", 0.6)),
                top_p=float(model_cfg.get("top_p", 0.95)),
                select_stop_prefill=select_stop_prefill,
                trace_separator=trace_separator,
                intermediate_stop=intermediate_stop,
                recovery_system=recovery_system,
                on_dialogue_complete=on_dialogue_complete,
                budget_deadline=budget_deadline,
                logprobs=capture_logprobs,
            )
        else:
            records, completed, failed, stopped_reason = run_dialogue_until_complete(
                states,
                backend,
                get_injection,
                adaptive_from=adaptive_from,
                intermediate_max_tokens=intermediate_max_tokens,
                stop_max_tokens=stop_max_tokens,
                recovery_max_tokens=recovery_max_tokens,
                temperature=float(model_cfg.get("temperature", 0.6)),
                top_p=float(model_cfg.get("top_p", 0.95)),
                select_stop_prefill=select_stop_prefill,
                on_dialogue_complete=on_dialogue_complete,
                budget_deadline=budget_deadline,
                injection_breaker=injection_breaker,
            )
    except InjectionCircuitBreakerError as exc:
        # Completed dialogues are already flushed per-dialogue; abort the rest.
        print(f"\n[ABORT] Injection circuit breaker tripped: {exc}", file=sys.stderr)
        print(
            "[ABORT] Completed dialogues were written and will be skipped on resume. "
            "Fix Anthropic credits/auth, then re-run the same command.",
            file=sys.stderr,
        )
        sys.exit(2)
    compute_elapsed_s = time.monotonic() - compute_t0
    total_elapsed_s = time.time() - progress_start

    print(
        f"\nDone in {total_elapsed_s:.1f}s (compute {compute_elapsed_s:.1f}s). "
        f"Dialogues: {dialogues_completed} | Output rows: {written} | "
        f"Skipped: {skipped} | Failed: {len(failed)} | stopped_reason: {stopped_reason}"
    )
    if failed:
        for st in failed[:5]:
            print(f"  FAIL {st.task['task_id']}: {st.fail_reason}")

    extrap = build_extrapolation(
        completed_dialogues=dialogues_completed,
        compute_elapsed_s=compute_elapsed_s,
        planned_tonight=planned_dialogues,
        scope=scope.get("name"),
    )
    if extrap.get("extrapolated_full_thesis_days") is not None:
        print(
            f"Extrapolation FULL THESIS ({extrap['planned_full_thesis']} dialogues): "
            f"~{extrap['extrapolated_full_thesis_days']} days "
            f"(~{extrap['extrapolated_full_thesis_h']}h, ~${extrap['extrapolated_full_cost_usd']} @ $1.36/h)"
        )

    if args.record_timings or args.budget_hours is not None:
        timings_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": "run_adaptive_dialogue.py",
            "scope": scope.get("name"),
            "model": model_name,
            "model_load_s": round(model_load_s, 1),
            "compute_elapsed_s": round(compute_elapsed_s, 1),
            "total_elapsed_s": round(total_elapsed_s, 1),
            "dialogues_completed": dialogues_completed,
            "dialogues_planned": planned_dialogues,
            "dialogues_skipped_existing": skipped_existing,
            "output_rows_written": written,
            "stopped_reason": stopped_reason,
            "budget_hours": args.budget_hours,
            "balanced_per_domain": args.balanced_per_domain,
            **extrap,
        }
        write_run_timings(timings_path, timings_payload)
        print(f"Timings: {timings_path}")

    write_run_metadata_summary(
        output_dir=timings_path.parent,
        scope_name=scope.get("name"),
        model_name=model_name,
        wall_time_s=total_elapsed_s,
        dialogues_completed=dialogues_completed,
        stopped_reason=stopped_reason,
        backend=backend,
        continuation_sampling=cont_sampling,
        is_continuation=is_continuation,
    )
    print(f"Output: {out_path}")
    print(f"Complete dialogues: {complete_path}")

    # Grid completeness check against the balanced selection actually run.
    missing_cells = check_grid_completeness(out_path, tasks, lists, model_name)

    # Do not let stopped_reason: complete mask dialogue-level failures: a non-zero
    # exit makes tmux/automation treat the run as failed so it is not assumed done.
    if failed:
        print(
            f"\n[FAIL] {len(failed)} dialogue(s) failed; exiting non-zero so the run "
            "is not mistaken for complete.",
            file=sys.stderr,
        )
        sys.exit(1)
    if missing_cells:
        print(
            f"\n[FAIL] {len(missing_cells)} grid cell(s) missing; exiting non-zero. "
            "Resume the run to fill them before Script 4.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
