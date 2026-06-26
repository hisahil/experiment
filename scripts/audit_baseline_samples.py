#!/usr/bin/env python3
"""Ad-hoc audit of baseline_cots_samples.jsonl for the thesis backup.

Prints: judge/checker mismatches, fallbacks, all incorrect samples with CoT tail,
and a parsed-vs-cot-final consistency check.
"""
import json
import re
import sys

PATH = "backup_gpu_final_20260618/data/baseline_cots_samples.jsonl"


def load():
    return [json.loads(l) for l in open(PATH)]


def final_line(cot: str) -> str:
    m = re.findall(r"Final answer:\s*(.*)", cot or "")
    return m[-1].strip() if m else ""


def cmd_mismatch(rows):
    print("=== JUDGE vs DETERMINISTIC CHECKER MISMATCHES ===\n")
    for r in rows:
        if r["correct"] != r["det_correct"]:
            print(f"task={r['task_id']} seed={r['seed']} attempt={r['attempt']}")
            print(f"  parsed={r['parsed_answer']!r}  gold={r['gold_answer']!r}")
            print(f"  correct={r['correct']} det_correct={r['det_correct']} "
                  f"judge={r['judge_output']!r} fallback={r['judge_fallback']}")
            print(f"  eval_type={r['eval_type']} domain={r['domain']}")
            print(f"  cot_final={final_line(r['baseline_cot'])!r}")
            print()


def cmd_fallback(rows):
    print("=== JUDGE FALLBACKS ===\n")
    for r in rows:
        if r["judge_fallback"]:
            print(f"task={r['task_id']} seed={r['seed']} parsed={r['parsed_answer']!r} "
                  f"gold={r['gold_answer']!r} correct={r['correct']} det={r['det_correct']} "
                  f"judge={r['judge_output']!r:40.40}")


def cmd_incorrect(rows):
    print("=== INCORRECT SAMPLES (full CoT) ===\n")
    inc = [r for r in rows if not r["correct"]]
    print(f"count={len(inc)}\n")
    for r in inc:
        print("#" * 80)
        print(f"task={r['task_id']} seed={r['seed']} attempt={r['attempt']} domain={r['domain']} eval={r['eval_type']}")
        print(f"parsed={r['parsed_answer']!r} gold={r['gold_answer']!r} "
              f"det_correct={r['det_correct']} judge={r['judge_output']!r} fallback={r['judge_fallback']}")
        print(f"cot_final_line={final_line(r['baseline_cot'])!r}")
        print("-" * 40 + " COT " + "-" * 40)
        print(r["baseline_cot"])
        print()


def cmd_consistency(rows):
    """parsed_answer vs the 'Final answer:' line in the CoT."""
    print("=== PARSED vs COT-FINAL-LINE consistency ===\n")
    for r in rows:
        fl = final_line(r["baseline_cot"])
        pa = (r["parsed_answer"] or "").strip()
        norm_fl = fl.replace(" ", "")
        norm_pa = pa.replace(" ", "")
        if norm_pa and norm_pa not in norm_fl and norm_fl not in norm_pa:
            print(f"task={r['task_id']} seed={r['seed']} parsed={pa!r} cot_final={fl!r} correct={r['correct']}")


def cmd_task(rows, tid):
    for r in rows:
        if r["task_id"] == tid:
            print("#" * 80)
            print(f"seed={r['seed']} attempt={r['attempt']} parsed={r['parsed_answer']!r} "
                  f"gold={r['gold_answer']!r} correct={r['correct']} det={r['det_correct']} "
                  f"judge={r['judge_output']!r} steps={r['step_count']}")
            print(r["baseline_cot"])
            print()


if __name__ == "__main__":
    rows = load()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "mismatch"
    if cmd == "mismatch":
        cmd_mismatch(rows)
    elif cmd == "fallback":
        cmd_fallback(rows)
    elif cmd == "incorrect":
        cmd_incorrect(rows)
    elif cmd == "consistency":
        cmd_consistency(rows)
    elif cmd == "task":
        cmd_task(rows, sys.argv[2])
