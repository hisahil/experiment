#!/usr/bin/env python3
"""Re-grade OLMo baseline samples with live Claude judge (no silent fallbacks).

Updates judge fields on the samples file and rebuilds majority / DeepSeek picker
rows in the baselines file. Does not regenerate model text.

Safety:
  - Judge model defaults to configs/experiment.yaml (claude-sonnet-4-6).
  - --limit never overwrites production paths; requires --out-samples/--out-baselines
    (or writes beside inputs with a .limitN suffix).
  - Full runs write atomically via .tmp then rename.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from answer_judge import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    JudgeCreditError,
    call_anthropic_judge,
    grade_sample,
    parse_judge_response,
    write_judge_raw_log,
)
from cot_segmentation import DEFAULT_RUNAWAY_MAX_STEPS, detect_degenerate_trace  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from evaluator import extract_domain_answer, extract_final_answer, gross_numeric_mismatch  # noqa: E402
from llm_backend import prepare_baseline_raw_text  # noqa: E402
from model_family import resolve_model_family  # noqa: E402
from reprocess_olmo_baselines import (  # noqa: E402
    ADJUDICATIONS,
    apply_adjudication,
    apply_seed_override,
    build_baseline_record,
    pick_baseline,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def resolve_judge_model(cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    cfg_path = ROOT / "configs/experiment.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        model = (cfg.get("judge") or {}).get("model")
        if model:
            return str(model)
    return DEFAULT_JUDGE_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/baseline_cots_samples.jsonl",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/baseline_cots.jsonl",
    )
    parser.add_argument("--out-samples", type=Path, default=None)
    parser.add_argument("--out-baselines", type=Path, default=None)
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Think")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Grade only first N tasks (writes to non-production outs).")
    parser.add_argument("--abort-on-fallback", action="store_true", default=True)
    parser.add_argument("--allow-fallback", action="store_true", help="Do not abort on judge_fallback.")
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=3,
        help="Extra judge re-calls when the response is unparseable (not Correct/Incorrect).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional JSONL checkpoint of graded samples for resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already present in --checkpoint with judge_fallback=False.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    family = resolve_model_family(args.model)
    majority = args.n_samples // 2 + 1
    judge_model = resolve_judge_model(args.judge_model)
    judge_cfg = {
        "provider": "anthropic",
        "model": judge_model,
        "enabled": True,
        "max_tokens": 32,
        "temperature": 0,
        "max_retries": 3,
    }
    abort_on_fallback = args.abort_on_fallback and not args.allow_fallback

    out_samples = args.out_samples
    out_baselines = args.out_baselines
    if args.limit is not None:
        if out_samples is None:
            out_samples = args.samples.with_name(
                args.samples.stem + f".limit{args.limit}.jsonl"
            )
        if out_baselines is None:
            out_baselines = args.baselines.with_name(
                args.baselines.stem + f".limit{args.limit}.jsonl"
            )
        if out_samples.resolve() == args.samples.resolve() or out_baselines.resolve() == args.baselines.resolve():
            raise SystemExit(
                "--limit refuses to overwrite production --samples/--baselines; "
                "pass distinct --out-samples and --out-baselines."
            )
    else:
        if out_samples is None:
            out_samples = args.samples
        if out_baselines is None:
            out_baselines = args.baselines

    prompts: dict[str, str] = {}
    tasks_path = ROOT / "data/experiment_tasks.jsonl"
    if tasks_path.exists():
        for row in load_jsonl(tasks_path):
            prompts[row["task_id"]] = row.get("prompt") or ""

    all_samples = load_jsonl(args.samples)
    olds = {r["task_id"]: r for r in load_jsonl(args.baselines)}
    order = list(olds) if olds else sorted({s["task_id"] for s in all_samples})
    if args.limit:
        order = order[: args.limit]
        keep = set(order)
        samples = [s for s in all_samples if s["task_id"] in keep]
    else:
        samples = list(all_samples)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = ROOT / "data/baselines/olmo-3-7b/.regrade_checkpoint.jsonl"
    done_keys: dict[tuple, dict] = {}
    if args.resume and checkpoint_path.exists():
        for row in load_jsonl(checkpoint_path):
            if row.get("judge_fallback"):
                continue
            key = (row.get("task_id"), row.get("seed"), row.get("attempt"))
            done_keys[key] = row
        print(f"[olmo-regrade] resume: loaded {len(done_keys)} graded samples from {checkpoint_path}")

    print(
        f"[olmo-regrade] samples={len(samples)} tasks={len({s['task_id'] for s in samples})} "
        f"judge={judge_model} out_baselines={out_baselines} out_samples={out_samples}"
    )
    t0 = time.monotonic()
    fallbacks = 0
    conflicts = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_mode = "a" if args.resume and checkpoint_path.exists() else "w"
    ckpt_fh = checkpoint_path.open(ckpt_mode, encoding="utf-8")

    def _one_word_retry() -> dict:
        """Force a Correct/Incorrect token when the first judge response is verbose."""
        prompt = (
            "Grade whether the student's extracted answer matches the reference.\n"
            f"Question: {question}\n"
            f"Reference answer: {gold}\n"
            f"Student extracted answer: {parsed}\n"
            "Respond with ONLY one word: Correct or Incorrect."
        )
        raw = call_anthropic_judge(
            prompt=prompt,
            model=judge_model,
            max_tokens=8,
            temperature=0,
            system=JUDGE_SYSTEM_PROMPT,
            max_retries=3,
        )
        parsed_v = parse_judge_response(raw)
        if parsed_v is None:
            return {
                "correct": None,
                "verifier_score": 0,
                "judge_output": raw.strip(),
                "judge_raw_output": raw.strip(),
                "det_correct": False,
                "judge_fallback": True,
            }
        # Preserve det_correct from a prior full grade_sample when available.
        return {
            "correct": parsed_v,
            "verifier_score": int(parsed_v),
            "judge_output": "Correct" if parsed_v else "Incorrect",
            "judge_raw_output": raw.strip(),
            "det_correct": bool(s.get("det_correct", False)),
            "judge_fallback": False,
            "one_word_retry": True,
        }

    def _grade_with_parse_retries() -> dict:
        last = None
        attempts = 1 + max(0, args.parse_retries)
        for attempt in range(attempts):
            if attempt == 0:
                grading = grade_sample(
                    question=question,
                    gold_answer=gold,
                    answer_content=cot,
                    extracted_answer=parsed,
                    domain=domain,
                    eval_type=eval_type,
                    judge_cfg=judge_cfg,
                    math_tolerance=1e-4,
                    science_tolerance=0.05,
                    dry_run=False,
                )
            else:
                grading = _one_word_retry()
                # Keep deterministic cross-check from the first full grade.
                if last is not None:
                    grading["det_correct"] = last.get("det_correct", False)
            last = grading
            if not grading.get("judge_fallback"):
                return grading
            out = str(grading.get("judge_output") or "")
            if out.startswith("fallback:"):
                return grading
            print(
                f"[olmo-regrade] unparseable judge on {s['task_id']} "
                f"seed={s.get('seed')} attempt={attempt + 1}/{attempts}; retrying"
            )
            time.sleep(0.4 * (attempt + 1))
        return last or grading

    try:
        for i, s in enumerate(samples, 1):
            key = (s.get("task_id"), s.get("seed"), s.get("attempt"))
            if key in done_keys:
                prev = done_keys[key]
                for fld in (
                    "parsed_answer",
                    "correct",
                    "det_correct",
                    "judge_fallback",
                    "judge_output",
                    "verifier_score",
                    "judge_raw_path",
                    "judge_det_conflict",
                    "degenerate",
                    "degenerate_reason",
                ):
                    if fld in prev:
                        s[fld] = prev[fld]
                if s.get("judge_det_conflict"):
                    conflicts += 1
                if i % 25 == 0:
                    print(
                        f"[olmo-regrade] … {i}/{len(samples)} fallbacks={fallbacks} "
                        f"conflicts={conflicts} (resume skips)"
                    )
                continue

            domain = s.get("domain") or "math"
            eval_type = s.get("eval_type") or "symbolic"
            gold = str(s.get("gold_answer", ""))
            cot = s.get("baseline_cot") or ""
            if not cot and s.get("raw_output"):
                cot = prepare_baseline_raw_text(
                    str(s["raw_output"]), think_tags=family.think_tags
                )
            parsed = s.get("parsed_answer")
            if parsed is None:
                parsed = extract_domain_answer(cot, domain, eval_type) or extract_final_answer(cot)
            question = prompts.get(s["task_id"], "")
            try:
                grading = _grade_with_parse_retries()
            except JudgeCreditError:
                raise
            s["parsed_answer"] = parsed
            s["correct"] = bool(grading["correct"])
            s["det_correct"] = grading["det_correct"]
            s["judge_fallback"] = grading["judge_fallback"]
            s["judge_output"] = grading["judge_output"]
            s["verifier_score"] = grading["verifier_score"]
            s["judge_raw_path"] = write_judge_raw_log(
                judge_cfg=judge_cfg,
                phase="olmo_baseline_regrade",
                task_id=s["task_id"],
                grading=grading,
                dry_run=False,
                sample_index=s.get("attempt"),
                seed=s.get("seed"),
                extracted_answer=parsed,
                gold_answer=gold,
            )
            if grading["judge_fallback"]:
                fallbacks += 1
                msg = (
                    f"[olmo-regrade] judge_fallback on {s['task_id']} "
                    f"seed={s.get('seed')}: {str(grading.get('judge_output'))[:500]}"
                )
                if abort_on_fallback:
                    raise SystemExit(msg)
                print(msg)
            judge_det_conflict = (
                eval_type == "symbolic"
                and not grading["judge_fallback"]
                and grading["correct"] is True
                and not grading["det_correct"]
            )
            sanity_gate_flipped = judge_det_conflict and gross_numeric_mismatch(
                parsed, gold
            )
            if sanity_gate_flipped:
                s["correct"] = False
            s["sanity_gate_flipped"] = sanity_gate_flipped
            s["judge_det_conflict"] = judge_det_conflict
            apply_adjudication(s)
            if s.get("judge_det_conflict"):
                conflicts += 1
            if "grading_mode" in s:
                del s["grading_mode"]
            steps = s.get("steps") or []
            if steps:
                deg, reason = detect_degenerate_trace(
                    steps, max_steps=DEFAULT_RUNAWAY_MAX_STEPS, raw_text=cot
                )
                s["degenerate"] = deg
                s["degenerate_reason"] = reason
            ckpt_fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            ckpt_fh.flush()
            if i % 25 == 0:
                print(f"[olmo-regrade] … {i}/{len(samples)} fallbacks={fallbacks} conflicts={conflicts}")
    finally:
        ckpt_fh.close()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        grouped[s["task_id"]].append(s)

    out_baseline_rows: list[dict] = []
    verified = 0
    for task_id in order:
        rows = grouped.get(task_id) or []
        rows.sort(key=lambda s: (s.get("attempt") or 0, s.get("seed") or 0))
        chosen, is_v, excl, degen_fb = pick_baseline(rows, majority, args.min_steps)
        chosen, is_v, excl, degen_fb, tier = apply_seed_override(
            rows, task_id, chosen, is_v, excl, degen_fb, majority, args.min_steps
        )
        old = olds.get(task_id) or {}
        if old.get("regenerated") and old.get("verified"):
            target = old.get("seed")
            match = next((r for r in rows if r.get("seed") == target and r.get("correct")), None)
            if match is not None and match.get("step_count", 0) >= args.min_steps:
                chosen = match
                is_v = True
                excl = None
                tier = old.get("selection_tier") or "targeted_regeneration_seed100"
        rec = build_baseline_record(
            task_id=task_id,
            old_record=old,
            samples=rows,
            chosen=chosen,
            verified=is_v,
            exclude_reason=excl,
            degenerate_fallback=degen_fb,
            n_samples=len(rows),
            selection_tier=tier,
        )
        if old.get("regenerated"):
            rec["regenerated"] = True
            rec["regen_base_seed"] = old.get("regen_base_seed", 100)
            rec["needs_regeneration"] = False
        out_baseline_rows.append(rec)
        if is_v:
            verified += 1

    out_sample_rows: list[dict] = []
    for tid in order:
        for s in sorted(grouped[tid], key=lambda x: (x.get("attempt") or 0, x.get("seed") or 0)):
            out_sample_rows.append(s)

    write_jsonl_atomic(out_baselines, out_baseline_rows)
    write_jsonl_atomic(out_samples, out_sample_rows)

    # Keep reprocessed mirrors in sync on full production writes.
    if out_baselines.resolve() == args.baselines.resolve() and args.limit is None:
        write_jsonl_atomic(
            args.baselines.with_name(args.baselines.stem + ".reprocessed.jsonl"),
            out_baseline_rows,
        )
        write_jsonl_atomic(
            args.samples.with_name(args.samples.stem + ".reprocessed.jsonl"),
            out_sample_rows,
        )

    dt = time.monotonic() - t0
    print(
        f"[olmo-regrade] done verified={verified}/{len(out_baseline_rows)} "
        f"fallbacks={fallbacks} conflicts={conflicts} wall={dt:.1f}s"
    )
    if conflicts:
        print(
            "[olmo-regrade] NOTE: judge/det conflicts present — review "
            "judge_det_conflict=True rows before treating as clean pool."
        )


if __name__ == "__main__":
    main()
