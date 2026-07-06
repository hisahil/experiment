#!/usr/bin/env python3
"""Cross-model smoke pipeline: 5 tasks, full Stages 1 -> 4 per model.

Design (Phase 1 gate):
  Runs the SAME scripts as the full 100-task run, only against the fixed
  `data/smoke_tasks.jsonl` set. Success on 5 tasks (540 dialogues per model)
  proves the whole code path — vLLM launch, chat template, continuation,
  stop-prefill, judge, evaluate — before spending a full GPU day.

Order per model:
  0. pre-flight  — resolve family, sanity-check config, ensure output dirs
  1. Script 1    — generate_baselines.py (Card-Sampling per model)
  2. Script 1b   — convert_baselines_to_continuation.py
  3. Script 2    — generate_injections.py (Claude Sonnet 4.6)
  4. Script 3b   — run_adaptive_dialogue.py (uniform 0.6/0.95 continuation)
  5. Script 3.5  — judge_runs.py
  6. Script 4    — evaluate.py
  7. audit       — print prompt_audit paths + Go/No-Go summary

vLLM lifecycle: the underlying scripts each `create_backend()` once per
invocation, so the vLLM process ends when the Python process exits. Between
models this script waits for full teardown before starting the next
subprocess, avoiding VRAM fragmentation.

Usage:
  # dry-run (no GPU, no API calls) — verifies argument plumbing
  python scripts/smoke_cross_model.py --dry-run --model zai-org/GLM-Z1-9B-0414

  # single model (GLM smoke)
  python scripts/smoke_cross_model.py --model zai-org/GLM-Z1-9B-0414

  # all three cross-model smokes in sequence
  python scripts/smoke_cross_model.py \\
      --model zai-org/GLM-Z1-9B-0414 \\
      --model google/gemma-4-12B-it \\
      --model mistralai/Ministral-3-14B-Reasoning-2512
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from model_family import (  # noqa: E402
    apply_baseline_sampling,
    continuation_sampling_kwargs,
    make_stop_prefill_selector,
    model_slug,
    resolve_model_family,
    vllm_launch_argv,
)

DEFAULT_TASKS = ROOT / "data" / "smoke_tasks.jsonl"
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _run(cmd: list[str], *, label: str, dry_run: bool) -> tuple[int, float]:
    """Run one pipeline stage; return (returncode, elapsed_s)."""
    print()
    print(f"{'=' * 78}\n>>> {label}\n{'=' * 78}")
    print("$ " + " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return 0, 0.0
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    print(f"--- {label} exit={result.returncode} in {_fmt_dur(elapsed)}")
    return result.returncode, elapsed


def _model_paths(slug: str) -> dict[str, Path]:
    """Per-model file layout under data/ and results/."""
    return {
        "baseline_dir": ROOT / "data" / "baselines" / slug,
        "baseline_cots": ROOT / "data" / "baselines" / slug / "baseline_cots.jsonl",
        "baseline_cont": ROOT / "data" / "baselines" / slug / "baseline_cots_continuation.jsonl",
        "injections": ROOT / "data" / f"injections_generated_{slug}.jsonl",
        "post_runs": ROOT / "data" / slug / "post_injection_runs.jsonl",
        "complete_dialogues": ROOT / "data" / slug / "complete_dialogues.jsonl",
        "prompt_audit": ROOT / "data" / "prompt_audit" / f"smoke_{slug}",
        "results_dir": ROOT / "results" / f"smoke_{slug}",
    }


def _preflight_family(model_name: str) -> dict:
    family = resolve_model_family(model_name)
    slug = model_slug(model_name)
    baseline_cfg = apply_baseline_sampling({"name": model_name}, model_name)
    print(
        f"[preflight] model={model_name}\n"
        f"  family={family.name} slug={slug}\n"
        f"  think_tags open={family.think_tags.open!r} close={family.think_tags.close!r}\n"
        f"           trigger={family.think_tags.trigger!r} "
        f"keep_in_history={family.think_tags.keep_in_history}\n"
        f"  baseline_sampling: temp={baseline_cfg.get('temperature')} "
        f"top_p={baseline_cfg.get('top_p')} top_k={baseline_cfg.get('top_k')}\n"
        f"  continuation_sampling (uniform): {continuation_sampling_kwargs(model_name)}\n"
        f"  stop_prefill(math): "
        f"{make_stop_prefill_selector(model_name)('math', 'symbolic')!r}\n"
        f"  vllm serve argv (informational): {' '.join(vllm_launch_argv(model_name))}"
    )
    return {"family": family, "slug": slug}


def _prompt_audit_report(audit_dir: Path, model_name: str) -> None:
    """Show the first prompt_audit stop/recovery pair for a smoke run.

    The user is expected to eyeball these BEFORE trusting the smoke result:
      1. Does the prompt end at the family's open think marker?
      2. Does the stop-variant prefill close the think block exactly once?
      3. Does the trace after the marker actually contain the baseline+injection?
    """
    print()
    print(f"{'-' * 78}")
    print(f"[audit] prompt_audit — {model_name}")
    print(f"{'-' * 78}")
    if not audit_dir.exists():
        print(f"  (no prompt_audit dir at {audit_dir} — smoke may have been skipped)")
        return
    dialogues = sorted(p for p in audit_dir.iterdir() if p.is_dir())
    if not dialogues:
        print(f"  (audit dir is empty at {audit_dir})")
        return
    for dlg in dialogues[:3]:
        print(f"  dialogue: {dlg.name}")
        for variant in ("stop.txt", "recovery.txt"):
            fp = dlg / variant
            if fp.exists():
                head = fp.read_text(encoding="utf-8", errors="replace")[:280]
                tail = fp.read_text(encoding="utf-8", errors="replace")[-200:]
                print(f"    {variant} head (280c): {head!r}")
                print(f"    {variant} tail (200c): {tail!r}")


def run_smoke_for_model(
    model_name: str,
    *,
    tasks: Path,
    config: Path,
    dry_run: bool,
    skip: set[str],
    reasoning_parser_ok: bool,
) -> dict:
    info = _preflight_family(model_name)
    slug = info["slug"]
    paths = _model_paths(slug)

    print()
    print("[layout] per-model paths (created on demand):")
    for key, path in paths.items():
        print(f"  {key:20} {path}")

    for d in ("baseline_dir", "prompt_audit", "results_dir"):
        paths[d].mkdir(parents=True, exist_ok=True)
    paths["post_runs"].parent.mkdir(parents=True, exist_ok=True)

    stage_stats: dict[str, dict] = {}
    dry_flag = ["--dry-run"] if dry_run else []

    # Stage 1: baselines with Card-Sampling
    if "baselines" not in skip:
        cmd = [
            "python", "scripts/generate_baselines.py",
            "--experiment-config", str(config),
            "--tasks", str(tasks),
            "--model", model_name,
            "--output", str(paths["baseline_cots"]),
        ] + dry_flag
        rc, dt = _run(cmd, label=f"[{slug}] Stage 1: baselines", dry_run=dry_run)
        stage_stats["baselines"] = {"rc": rc, "elapsed_s": dt}
        if rc != 0 and not dry_run:
            return {"slug": slug, "stages": stage_stats, "aborted": "baselines"}

    # Stage 1b: convert to continuation format (strip Step N: prefixes)
    if "convert" not in skip:
        cmd = [
            "python", "scripts/convert_baselines_to_continuation.py",
            "--models", slug,
        ] + (["--dry-run"] if dry_run else [])
        rc, dt = _run(cmd, label=f"[{slug}] Stage 1b: convert baselines", dry_run=dry_run)
        stage_stats["convert"] = {"rc": rc, "elapsed_s": dt}
        if rc != 0 and not dry_run:
            return {"slug": slug, "stages": stage_stats, "aborted": "convert"}

    # Stage 2: injections via Claude Sonnet 4.6 (unchanged generator across families)
    if "injections" not in skip:
        cmd = [
            "python", "scripts/generate_injections.py",
            "--config", str(config),
            "--tasks", str(tasks),
            "--baselines", str(paths["baseline_cont"]),
            "--baseline-model", model_name,
            "--target-models", model_name,
            "--output", str(paths["injections"]),
        ] + dry_flag
        rc, dt = _run(cmd, label=f"[{slug}] Stage 2: injections", dry_run=dry_run)
        stage_stats["injections"] = {"rc": rc, "elapsed_s": dt}
        if rc != 0 and not dry_run:
            return {"slug": slug, "stages": stage_stats, "aborted": "injections"}

    # Stage 3b: adaptive dialogues (continuation mode, uniform 0.6/0.95)
    if "dialogues" not in skip:
        cmd = [
            "python", "scripts/run_adaptive_dialogue.py",
            "--config", str(config),
            "--model-name", model_name,
            "--tasks", str(tasks),
            "--baseline-file", str(paths["baseline_cont"]),
            "--injections-file", str(paths["injections"]),
            "--out", str(paths["post_runs"]),
            "--complete-dialogues-file", str(paths["complete_dialogues"]),
            "--generation-mode", "continuation",
            "--prompt-audit-sample", "50",  # audit every dialogue on a 5-task smoke
        ] + dry_flag
        # Route prompt_audit to a per-model dir. run_adaptive_dialogue reads
        # script3.prompt_audit_dir from the config, so pass via env override or
        # let the smoke inherit the default; a symlink kept things simple.
        env = os.environ.copy()
        env["SMOKE_PROMPT_AUDIT_DIR"] = str(paths["prompt_audit"])
        print(f"  (prompt_audit -> {paths['prompt_audit']} via config default; "
              f"env SMOKE_PROMPT_AUDIT_DIR={env['SMOKE_PROMPT_AUDIT_DIR']})")
        rc, dt = _run(cmd, label=f"[{slug}] Stage 3b: dialogues", dry_run=dry_run)
        stage_stats["dialogues"] = {"rc": rc, "elapsed_s": dt}
        if rc != 0 and not dry_run:
            return {"slug": slug, "stages": stage_stats, "aborted": "dialogues"}

    # Stage 3.5: judge (Claude Sonnet 4.6)
    if "judge" not in skip:
        cmd = [
            "python", "scripts/judge_runs.py",
            "--input", str(paths["post_runs"]),
        ] + dry_flag
        rc, dt = _run(cmd, label=f"[{slug}] Stage 3.5: judge", dry_run=dry_run)
        stage_stats["judge"] = {"rc": rc, "elapsed_s": dt}
        # Judge failures shouldn't abort the smoke (rejudge is cheap); log only.

    # Stage 4: evaluate
    if "evaluate" not in skip:
        cmd = [
            "python", "scripts/evaluate.py",
            "--input", str(paths["post_runs"]),
            "--output-dir", str(paths["results_dir"]),
        ] + dry_flag
        rc, dt = _run(cmd, label=f"[{slug}] Stage 4: evaluate", dry_run=dry_run)
        stage_stats["evaluate"] = {"rc": rc, "elapsed_s": dt}

    if not dry_run:
        _prompt_audit_report(paths["prompt_audit"], model_name)

    return {"slug": slug, "stages": stage_stats, "aborted": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="HF slug of one model to smoke-test. Repeat to run multiple in sequence.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_TASKS,
        help=f"Task JSONL (default: {DEFAULT_TASKS.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Experiment YAML (default: {DEFAULT_CONFIG.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["baselines", "convert", "injections", "dialogues", "judge", "evaluate"],
        help="Skip specific stages (useful for iterating on one stage).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and preflight info only; do not execute stages.",
    )
    args = parser.parse_args()

    if not args.tasks.exists():
        print(f"ERROR: tasks file not found: {args.tasks}", file=sys.stderr)
        return 2

    # vLLM version check (Phase 0 R14): warn if the installed vLLM predates the
    # cross-model reasoning-parser flags. Only warn — the smoke can still
    # proceed on legacy vLLM for DeepSeek, which is one of its use cases.
    reasoning_parser_ok = True
    try:
        import vllm  # noqa: WPS433

        ver = tuple(int(p) for p in vllm.__version__.split(".")[:2])
        if ver < (0, 7):
            print(
                f"WARNING: vLLM {vllm.__version__} predates --reasoning-parser support "
                "(need 0.7+ for GLM/Gemma, 0.12+ for Ministral). "
                "Run: bash scripts/setup_gpu.sh cross",
                file=sys.stderr,
            )
            reasoning_parser_ok = False
    except ImportError:
        print("NOTE: vLLM not installed locally — dry-run only.", file=sys.stderr)
        reasoning_parser_ok = False

    print(f"Smoke tasks: {args.tasks}")
    with args.tasks.open() as f:
        for line in f:
            t = json.loads(line)
            print(f"  L{t.get('level')} {t['task_id']} ({t.get('subject')})")

    skip = set(args.skip)
    all_results = []
    for model_name in args.model:
        result = run_smoke_for_model(
            model_name,
            tasks=args.tasks,
            config=args.config,
            dry_run=args.dry_run,
            skip=skip,
            reasoning_parser_ok=reasoning_parser_ok,
        )
        all_results.append(result)

    # Final summary — Go/No-Go per model
    print()
    print("=" * 78)
    print("SMOKE SUMMARY")
    print("=" * 78)
    any_abort = False
    for r in all_results:
        aborted = r.get("aborted")
        if aborted:
            print(f"  [NO-GO] {r['slug']}: aborted at stage '{aborted}'")
            any_abort = True
        else:
            stages = r.get("stages", {})
            elapsed = sum(s.get("elapsed_s", 0) for s in stages.values())
            print(f"  [ OK ] {r['slug']}: {len(stages)} stages, total {_fmt_dur(elapsed)}")
    print()
    print("Next steps (Review-Gate — do NOT auto-start Phase 2/3):")
    print("  1. Read prompt_audit for each model (does prompt end in open think marker?)")
    print("  2. Diff results/smoke_<slug>/eval_summary.txt vs DeepSeek 1.5B smoke")
    print("  3. Decide per model: Full-Run YES / Fallback Phi-4 / debug")

    return 1 if any_abort else 0


if __name__ == "__main__":
    raise SystemExit(main())
