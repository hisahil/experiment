#!/usr/bin/env python3
"""Run Experiment 1 pipeline stages 0–4 with optional grid validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_common import build_stage_commands, print_plan, run_command  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1 pipeline (stages 0–4).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=ROOT / "configs/datasets.yaml",
    )
    parser.add_argument("--scope", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--balanced-per-domain", type=int, default=None)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--record-timings", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only (no subprocess).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--script3",
        choices=["three_turn", "adaptive"],
        default=None,
        help="Override script3.dialogue_mode (adaptive -> run_adaptive_dialogue.py).",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Start at this stage (0=prepare_tasks, 5=validate only with --validate-only).",
    )
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument(
        "--stage-a",
        action="store_true",
        help=(
            "Stage A API cost optimisation (adaptive mode only): Stage 2 as one "
            "Anthropic Message Batch with step1-only (--batch --step1-only), Stage 3 "
            "with inline judge disabled (--no-judge), plus a downstream judge_batch "
            "stage. Set ANTHROPIC_PROMPT_CACHE=1 to also cache the judge prefix."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validate_grid only (implies --from-stage 5).",
    )
    args = parser.parse_args()

    script3_override = None
    if args.script3 == "adaptive":
        script3_override = "adaptive_alternating"
    elif args.script3 == "three_turn":
        script3_override = "three_turn"

    from_stage = 5 if args.validate_only else args.from_stage
    skip_evaluate = args.skip_evaluate or args.validate_only

    stages = build_stage_commands(
        experiment_config=args.config,
        datasets_config=args.datasets_config,
        scope=args.scope,
        domain=args.domain,
        limit=args.limit,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        script3_override=script3_override,
        from_stage=from_stage,
        skip_evaluate=skip_evaluate,
        skip_validate=args.skip_validate and not args.validate_only,
        balanced_per_domain=args.balanced_per_domain,
        budget_hours=args.budget_hours,
        record_timings=args.record_timings,
        stage_a=args.stage_a,
    )

    if not stages:
        print("No stages selected.", file=sys.stderr)
        sys.exit(1)

    print_plan(stages)
    for stage_id, name, command in stages:
        print(f"\n=== Stage {stage_id}: {name} ===")
        run_command(command, dry_run=args.dry_run, record_timings=args.record_timings)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
