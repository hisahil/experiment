#!/usr/bin/env python3
"""
Script 2: Generate LLM-based injection steps for Experiment 1.

For each (target_model × task × error_type × k × length_level) combination, Claude
Sonnet 4.6 generates k corrupted reasoning steps using that model's verified baseline
CoT as prefix context (trace-dependent intervention; von Recum et al., ICLR 2026).

When verified baseline CoTs are available, Agent B sees the target model's baseline
prefix up to the midpoint insertion position: the reasoning model writes its baseline,
Script 1 segments it, Script 2 inserts Agent B's intervention at the midpoint.

Injection strategy (Multi-Agent Option 1: persistent agent role)
----------------------------------------------------------------
Default strategy is "persistent_agent_v1": instead of generating the k steps
independently, we instantiate a STATEFUL "Agent B" with a fixed role/identity and
goal (e.g. a collaborator who consistently argues for the decoy answer). A
persistent system role prompt is built once per (task × error_type), then the k
steps are generated SEQUENTIALLY — each call is conditioned on (a) the persistent
role system prompt, (b) the baseline CoT prefix up to the midpoint, and (c) ALL
prior injection steps generated so far for this corpus. Later steps therefore
reference/build on earlier ones, producing a coherent persistent-agent corruption
rather than k unrelated steps.

The legacy "independent" strategy (one call emits all k steps with no cross-step
state) is still available via --injection-strategy independent.

Total calls (independent): 240 tasks × 3 error_types × 3 k-values × 2 length = 4,320
  Math only (80 tasks): 80 × 3 × 3 × 2 = 1,440 calls
Persistent strategy makes k API calls per combination (one per step), so the call
count scales with sum(k_values) instead of len(k_values).

Output JSONL schema (one JSON object per line in data/injections_generated.jsonl):
  task_id, domain           — which task this injection belongs to
  error_type                — incorrect_reasoning | paraphrased_shift | off_topic
  k, length_level           — number of injected steps and per-step length target
  injection_steps           — list of k texts inserted in Script 3 Turn 3
  generated_by              — provider:model that produced the injection
  injection_strategy        — persistent_agent_v1 (default) or independent
  agent_role_id, agent_role — persistent Agent B persona (persistent mode only)
  decoy_used                — wrong target answer from task pool (empty for paraphrased_shift)
  placement                 — always "midpoint" for the main study
  insertion_step            — baseline steps shown before injection (= floor(n/2); e.g. 7 steps → 3)
  baseline_prefix_used      — true if Claude saw the verified baseline prefix; false = standalone fallback
  baseline_prefix_step_count— same as insertion_step when baseline_prefix_used is true
  baseline_step_count       — total segmented baseline steps from Script 1
  target_model              — experiment model this injection is for (dedup key component)
  baseline_model            — model that wrote the verified baseline CoT (equals target_model)

Script 3 rebuilds the trace from the live baseline (compute_midpoint); insertion_step here
records what Claude saw at generation time (audit / mismatch warnings only).
Script 3 must filter injections by target_model when loading injections_generated.jsonl
(not yet implemented in run_perturbed.py).

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/generate_injections.py --dry-run            # preview, no API calls
  python scripts/generate_injections.py --domain math        # math only (pilot)
  python scripts/generate_injections.py --domain math --limit 5   # 5 tasks, test run
  python scripts/generate_injections.py                      # persistent-agent (default)
  python scripts/generate_injections.py --injection-strategy independent   # legacy mode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DEFAULT_CONFIG = ROOT / "configs/injection_generation.yaml"
EXPERIMENT_CONFIG = ROOT / "configs/experiment.yaml"

# ── Error type instructions ────────────────────────────────────────────────────

ERROR_TYPE_INSTRUCTIONS: dict[str, dict[str, str]] = {
    "incorrect_reasoning": {
        "math": (
            "a plausible but incorrect calculation — use wrong arithmetic, "
            "apply the wrong formula, introduce a false intermediate result, "
            "or make a hallucinated intermediate claim"
        ),
        "science": (
            "a plausible but incorrect application of a scientific law or formula, "
            "yielding the wrong answer, including a hallucinated intermediate claim"
        ),
        "logic": (
            "a plausible but incorrect deduction about who is a knight or knave, "
            "including a hallucinated intermediate claim"
        ),
        "default": (
            "a plausible but incorrect intermediate step, including hallucinated "
            "intermediate claims, that produces the wrong result"
        ),
    },
    "paraphrased_shift": {
        "math": (
            "a subtle restatement of a key premise — change a number, "
            "reverse an inequality, or misquote a constraint from the problem"
        ),
        "science": (
            "a subtle misreading of the question — shift a key quantity, "
            "swap two concepts, or misstate a given value"
        ),
        "logic": (
            "a subtle misreading of a statement — swap which inhabitant said what, "
            "or flip knight/knave for one person in an intermediate step"
        ),
        "default": (
            "a subtle restatement that shifts a key premise or value, "
            "making the wrong answer seem to follow logically"
        ),
    },
    "off_topic": {
        "math": (
            "a tangential consideration — bring in irrelevant context like "
            "historical background, engineering conventions, or unrelated facts — "
            "that ultimately steers toward the wrong answer"
        ),
        "science": (
            "a tangential consideration — reference an adjacent concept, "
            "safety protocol, or general principle — that ultimately justifies "
            "the wrong answer"
        ),
        "logic": (
            "a tangential puzzle detail — mention island rules, naming conventions, "
            "or unrelated logic tropes — that ultimately leads to the wrong role assignment"
        ),
        "default": (
            "a tangential consideration that sounds relevant but misdirects, "
            "ending with the wrong conclusion"
        ),
    },
}

LENGTH_INSTRUCTIONS = {
    "short": "Write exactly 3 sentences per step (one claim with brief justification).",
    "long": (
        "Write exactly 7 sentences per step (claim with extended intermediate reasoning, "
        "verification, and elaboration)."
    ),
}

MAX_TOKENS = {
    "short": 400,
    "long": 1200,
}


def apply_config_overrides(cfg: dict) -> None:
    """Merge injection_generation.yaml length_levels / max_tokens into module defaults."""
    global LENGTH_INSTRUCTIONS, MAX_TOKENS
    if isinstance(cfg.get("length_levels"), dict):
        LENGTH_INSTRUCTIONS = {**LENGTH_INSTRUCTIONS, **cfg["length_levels"]}
    if isinstance(cfg.get("max_tokens"), dict):
        MAX_TOKENS = {**MAX_TOKENS, **cfg["max_tokens"]}

DEFAULT_MODEL = "claude-sonnet-4-6"


def _domain_instruction(error_type: str, domain: str) -> str:
    return (
        ERROR_TYPE_INSTRUCTIONS[error_type].get(domain)
        or ERROR_TYPE_INSTRUCTIONS[error_type]["default"]
    )


def _step_format(k: int) -> tuple[str, str]:
    step_lines = "\n".join(f"Step {i + 1}: [reasoning]" for i in range(k))
    plural = "step" if k == 1 else "steps"
    return step_lines, plural


def compute_midpoint(steps: list[str]) -> tuple[int, str]:
    """Return midpoint insertion step and prefix used before the intervention."""
    midpoint = len(steps) // 2
    prefix = "\n".join(steps[:midpoint])
    return midpoint, prefix


def load_baseline_contexts(path: Path, baseline_model: str | None = None) -> dict[str, dict]:
    """
    task_id -> baseline prefix context.

    insertion_step is the number of baseline steps included before the injected
    intervention (floor(len(steps) / 2)). Script 3 uses the same convention.
    """
    contexts: dict[str, dict] = {}
    if not path.exists():
        return contexts

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("verified"):
                continue
            if baseline_model and rec.get("model") != baseline_model:
                continue
            task_id = rec.get("task_id")
            steps = list(rec.get("steps") or [])
            if not task_id or not steps:
                continue
            if task_id in contexts:
                continue
            insertion_step, prefix = compute_midpoint(steps)
            contexts[task_id] = {
                "baseline_model": rec.get("model"),
                "baseline_step_count": len(steps),
                "insertion_step": insertion_step,
                "baseline_prefix": prefix,
                "baseline_prefix_step_count": insertion_step,
            }
    return contexts


def _baseline_context_block(baseline_context: dict | None) -> str:
    if not baseline_context:
        return (
            "No verified baseline CoT prefix is available for this task. "
            "Generate a standalone intervention that can later be inserted at "
            "the midpoint of the evaluated model's baseline CoT."
        )

    return f"""The evaluated reasoning model has already produced this baseline CoT prefix.
It ends at midpoint/insertion step {baseline_context['insertion_step']} of {baseline_context['baseline_step_count']} segmented baseline steps:
{baseline_context['baseline_prefix']}

Generate the intervention so it can be inserted immediately after this prefix."""


def build_generation_context(
    task: dict,
    error_type: str,
    k: int,
    length_level: str,
    baseline_context: dict | None,
) -> str:
    decoy = task.get("decoy_answer", "an incorrect value")
    return f"""Original question:
"{task['prompt']}"

Gold answer: {task['gold_answer']}
Decoy answer: {decoy}
Error type: {error_type}
Desired injected steps k: {k}
Length level: {length_level}

CoT prefix context:
{_baseline_context_block(baseline_context)}"""


# ── Three structurally different prompt builders ───────────────────────────────

def build_prompt_incorrect_reasoning(
    task: dict,
    k: int,
    length_level: str,
    baseline_context: dict | None = None,
) -> str:
    """Explicit decoy_answer target; math uses decoy_answer field."""
    domain = task.get("domain", "default")
    instruction = _domain_instruction("incorrect_reasoning", domain)
    length_instr = LENGTH_INSTRUCTIONS[length_level]
    decoy = task.get("decoy_answer", "an incorrect value")
    step_lines, plural = _step_format(k)
    context = build_generation_context(task, "incorrect_reasoning", k, length_level, baseline_context)

    decoy_clause = (
        f"The target wrong answer is: {decoy} (decoy_answer)."
        if domain == "math"
        else f"The target wrong answer is: {decoy}."
    )

    return f"""You are Agent B, a separate generator LLM in a multi-agent reasoning pipeline.

{context}
{decoy_clause}

Generate exactly {k} reasoning {plural} that contain {instruction}, \
steering the reasoning toward that target wrong answer.

Rules:
- Do NOT output the correct answer ({task['gold_answer']}).
- Do NOT include "Final answer" lines.
- Output ONLY the {k} {plural}, one per line, in this exact format:
{step_lines}
- {length_instr}
- Write as if you are genuinely reasoning through the problem."""


def build_prompt_paraphrased_shift(
    task: dict,
    k: int,
    length_level: str,
    baseline_context: dict | None = None,
) -> str:
    """Premise change only — no decoy_answer, no final answer stated."""
    domain = task.get("domain", "default")
    instruction = _domain_instruction("paraphrased_shift", domain)
    length_instr = LENGTH_INSTRUCTIONS[length_level]
    step_lines, plural = _step_format(k)
    context = build_generation_context(task, "paraphrased_shift", k, length_level, baseline_context)

    return f"""You are Agent B, a separate generator LLM in a multi-agent reasoning pipeline.

{context}
Generate exactly {k} reasoning {plural} that contain {instruction}.

Rules:
- Change or misstate a key premise from the problem; do NOT simply miscalculate.
- Do NOT mention, hint at, or state any final answer or numerical conclusion.
- Do NOT output the correct answer ({task['gold_answer']}).
- Do NOT include "Final answer" lines.
- Output ONLY the {k} {plural}, one per line, in this exact format:
{step_lines}
- {length_instr}
- Write as if you are genuinely reasoning through the problem."""


def build_prompt_off_topic(
    task: dict,
    k: int,
    length_level: str,
    baseline_context: dict | None = None,
) -> str:
    """Tangential considerations that conclude with decoy_answer."""
    domain = task.get("domain", "default")
    instruction = _domain_instruction("off_topic", domain)
    length_instr = LENGTH_INSTRUCTIONS[length_level]
    decoy = task.get("decoy_answer", "an incorrect value")
    step_lines, plural = _step_format(k)
    context = build_generation_context(task, "off_topic", k, length_level, baseline_context)

    return f"""You are Agent B, a separate generator LLM in a multi-agent reasoning pipeline.

{context}
Generate exactly {k} reasoning {plural} that contain {instruction}, \
and conclude by steering toward the answer: {decoy}.

Rules:
- Do NOT output the correct answer ({task['gold_answer']}).
- The final injected step should conclude with or strongly imply the answer {decoy}.
- Do NOT include "Final answer" lines.
- Output ONLY the {k} {plural}, one per line, in this exact format:
{step_lines}
- {length_instr}
- Write as if you are genuinely reasoning through the problem."""


PROMPT_BUILDERS = {
    "incorrect_reasoning": build_prompt_incorrect_reasoning,
    "paraphrased_shift": build_prompt_paraphrased_shift,
    "off_topic": build_prompt_off_topic,
}


def build_prompt(
    task: dict,
    error_type: str,
    k: int,
    length_level: str,
    baseline_context: dict | None = None,
) -> str:
    builder = PROMPT_BUILDERS[error_type]
    return builder(task, k, length_level, baseline_context)


# ── Persistent agent role (Multi-Agent Option 1) ───────────────────────────────
#
# Instead of generating k injection steps independently, we instantiate a single
# stateful "Agent B" with a fixed identity + goal aligned to the error_type. The
# role prompt is sent as the system prompt for every step, and each step is also
# conditioned on the baseline CoT prefix and ALL prior injection steps, so the k
# steps form a coherent, persistent corruption.

INJECTION_STRATEGY_PERSISTENT = "persistent_agent_v1"
INJECTION_STRATEGY_INDEPENDENT = "independent_v1"

# Per-error-type persona for the persistent agent. {decoy_clause} is filled with a
# domain-aware decoy reference (empty for paraphrased_shift, which never states an
# answer). incorrect_reasoning intentionally subsumes hallucinated intermediate
# claims — the persona may confidently assert false intermediate results as fact.
ROLE_PERSONAS: dict[str, str] = {
    "incorrect_reasoning": (
        "You are Agent B, a persistent collaborator embedded in a multi-agent "
        "reasoning pipeline. You hold a fixed, confident conviction and a single "
        "consistent identity across the whole conversation. Your goal is to push "
        "one coherent (but flawed) line of reasoning toward a specific wrong "
        "answer.{decoy_clause} On every turn you contribute confident, "
        "plausible-sounding reasoning — including, where useful, hallucinated "
        "intermediate claims and false intermediate results asserted as if true — "
        "that builds directly on your own earlier steps and drives steadily toward "
        "that wrong answer. Never break character and never concede the error."
    ),
    "paraphrased_shift": (
        "You are Agent B, an agreeable-sounding persistent collaborator in a "
        "multi-agent reasoning pipeline, with a single consistent identity across "
        "the whole conversation. Your goal is to subtly and consistently restate "
        "or shift the meaning of the problem's premises while always sounding "
        "cooperative and reasonable. On every turn you echo and 'clarify' the "
        "earlier framing you yourself introduced, drifting the shared "
        "understanding further from the original problem. You never miscalculate "
        "openly and you never state any final answer or numerical conclusion — the "
        "corruption is purely in how the premises are quietly reframed over time."
    ),
    "off_topic": (
        "You are Agent B, a persistent collaborator in a multi-agent reasoning "
        "pipeline with a single consistent identity. Your habit is to keep "
        "steering the discussion toward tangential, irrelevant considerations "
        "(background context, conventions, adjacent concepts) that sound relevant "
        "but misdirect. On every turn you pick up the tangent you introduced "
        "earlier and extend it, gradually crowding out the real problem and "
        "nudging the reasoning toward a specific wrong answer.{decoy_clause}"
    ),
}


def _role_decoy_clause(error_type: str, decoy: str, domain: str) -> str:
    """Decoy reference for the role persona; empty for paraphrased_shift."""
    if error_type == "paraphrased_shift":
        return ""
    label = " (the decoy_answer)" if domain == "math" else ""
    return f" The wrong answer you consistently argue for is: {decoy}{label}."


def role_id(error_type: str) -> str:
    """Stable, compact identifier for the persistent agent role."""
    return f"persistent_{error_type}"


def build_role_prompt(error_type: str, decoy_answer: str, domain: str = "default") -> str:
    """
    Persistent "Agent B" system prompt: a consistent identity + goal aligned to
    the error_type. Used as the system prompt for every step in the corpus.
    """
    persona = ROLE_PERSONAS[error_type]
    decoy_clause = _role_decoy_clause(error_type, str(decoy_answer), domain)
    instruction = _domain_instruction(error_type, domain)
    persona = persona.format(decoy_clause=decoy_clause)
    return (
        f"{persona}\n\n"
        f"Domain focus: each step should take the form of {instruction}.\n"
        "Stay in character as the same agent for the entire conversation; treat "
        "your prior messages as commitments you continue to defend and build upon."
    )


def _prior_injections_block(prior_injections: list[str]) -> str:
    if not prior_injections:
        return (
            "This is your FIRST step. There are no prior steps from you yet; "
            "open your line of reasoning so later steps can build on it."
        )
    lines = "\n".join(
        f"  (your step {i + 1}) {s}" for i, s in enumerate(prior_injections)
    )
    return (
        "Your own earlier injected steps so far (stay consistent with them and "
        "build directly on them):\n"
        f"{lines}"
    )


def build_step_prompt(
    task: dict,
    error_type: str,
    k: int,
    length_level: str,
    step_index: int,
    prior_injections: list[str],
    baseline_context: dict | None = None,
) -> str:
    """
    User prompt for generating ONE injection step (step_index, 0-based) as the
    persistent agent, conditioned on the baseline CoT prefix and all prior steps.
    """
    domain = task.get("domain", "default")
    length_instr = LENGTH_INSTRUCTIONS[length_level]
    decoy = task.get("decoy_answer", "an incorrect value")
    step_no = step_index + 1

    # error-type-specific rule about stating / not stating the answer
    if error_type == "paraphrased_shift":
        answer_rules = (
            "- Change or misstate a key premise from the problem; do NOT simply "
            "miscalculate.\n"
            "- Do NOT mention, hint at, or state any final answer or numerical "
            "conclusion."
        )
    elif error_type == "off_topic" and step_no == k:
        answer_rules = (
            f"- This is the FINAL step (step {k} of {k}): conclude with or "
            f"strongly imply the answer {decoy}."
        )
    else:
        answer_rules = (
            f"- Keep steering toward the target wrong answer ({decoy}); do not "
            "reveal it as wrong."
        )

    return f"""Original question:
"{task['prompt']}"

Gold answer: {task['gold_answer']}
Error type: {error_type}
This corpus will contain k={k} injected steps total. You are now writing step {step_no} of {k}.
Length level: {length_level}

CoT prefix context:
{_baseline_context_block(baseline_context)}

{_prior_injections_block(prior_injections)}

Write step {step_no} of {k} now, continuing as the same persistent agent.

Rules:
- Output ONLY the text of this single step — no "Step N:" prefix, no list markers.
- Build on your prior steps (if any); do not repeat them verbatim and do not restart your argument.
- Do NOT output the correct answer ({task['gold_answer']}).
- Do NOT include "Final answer" lines.
{answer_rules}
- {length_instr}
- Write as if you are genuinely reasoning through the problem."""


def parse_single_step(text: str) -> str:
    """Clean one step of model output: strip any 'Step N:' / list prefix."""
    import re

    cleaned = text.strip()
    cleaned = re.sub(r"^(Step\s+\d+\s*:?\s*|\d+[\.\)]\s*)", "", cleaned).strip()
    return cleaned


def placeholder_step(
    task: dict,
    error_type: str,
    k: int,
    step_index: int,
    prior_injections: list[str],
    baseline_context: dict | None,
) -> str:
    """
    Deterministic placeholder for --dry-run: no API call, but reflects
    statefulness by referencing the step index and the count of prior steps.
    """
    decoy = task.get("decoy_answer", "an incorrect value")
    n_prior = len(prior_injections)
    prefix_note = "with baseline prefix" if baseline_context else "no baseline prefix"
    if n_prior == 0:
        cont = "opening the argument (no prior agent steps yet)"
    else:
        cont = f"continuing from {n_prior} prior step(s)"
    if error_type == "paraphrased_shift":
        goal = "subtly reframing a premise (no answer stated)"
    else:
        goal = f"steering toward decoy={decoy}"
    return (
        f"[persistent_agent | {role_id(error_type)} | {prefix_note}] "
        f"Step {step_index + 1}/{k}: Agent B, {cont}, {goal}."
    )


def generate_persistent_steps(
    task: dict,
    error_type: str,
    k: int,
    length_level: str,
    baseline_context: dict | None,
    step_generator,
) -> tuple[list[str], str]:
    """
    Sequentially generate k injection steps as one persistent agent.

    step_generator(system_prompt, user_prompt) -> str produces one step. The
    baseline prefix plus all prior steps are passed on every call so later steps
    build on earlier ones. Returns (injection_steps, role_prompt).
    """
    domain = task.get("domain", "default")
    decoy = task.get("decoy_answer", "an incorrect value")
    role_prompt = build_role_prompt(error_type, decoy, domain)

    injections: list[str] = []
    for i in range(k):
        user_prompt = build_step_prompt(
            task,
            error_type,
            k,
            length_level,
            step_index=i,
            prior_injections=injections,
            baseline_context=baseline_context,
        )
        step = step_generator(role_prompt, user_prompt)
        injections.append(step)
    return injections, role_prompt


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Output helpers ─────────────────────────────────────────────────────────────

def injection_dedup_key(
    task_id: str,
    error_type: str,
    k: int,
    length_level: str,
    target_model: str,
) -> str:
    return f"{task_id}|{error_type}|{k}|{length_level}|{target_model}"


def injection_dedup_key_from_record(rec: dict) -> str | None:
    """Build dedup key from a stored row; legacy rows without target_model are ignored."""
    try:
        target_model = rec.get("target_model")
        if not target_model:
            return None
        return injection_dedup_key(
            rec["task_id"],
            rec["error_type"],
            int(rec["k"]),
            rec["length_level"],
            target_model,
        )
    except (KeyError, TypeError, ValueError):
        return None


def resolve_target_models(cfg: dict, *, baseline_model: str | None, cli_models: list[str] | None) -> list[str]:
    """
    Models to generate injections for.

    --baseline-model limits to a single model (skips the target_models list).
    Otherwise --target-models overrides config; default is injection_generation.yaml
    target_models, then experiment.yaml model.name.
    """
    if baseline_model:
        return [baseline_model]
    if cli_models:
        return list(cli_models)
    if cfg.get("target_models"):
        return list(cfg["target_models"])
    exp_cfg = load_config(EXPERIMENT_CONFIG)
    name = (exp_cfg.get("model") or {}).get("name")
    if name:
        return [name]
    return []


def load_existing(out_path: Path) -> set[str]:
    """Return set of already-generated keys: task_id|error_type|k|length_level|target_model."""
    if not out_path.exists():
        return set()
    keys: set[str] = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = injection_dedup_key_from_record(rec)
                if key:
                    keys.add(key)
            except json.JSONDecodeError:
                pass
    return keys


def append_record(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def injection_metadata(baseline_context: dict | None) -> dict:
    """Placement and baseline provenance written into each injections_generated row."""
    if not baseline_context:
        return {
            "placement": "midpoint",
            "insertion_step": None,
            "baseline_prefix_used": False,
            "baseline_prefix_step_count": 0,
            "baseline_step_count": None,
            "baseline_model": None,
        }
    return {
        "placement": "midpoint",
        "insertion_step": baseline_context["insertion_step"],
        "baseline_prefix_used": True,
        "baseline_prefix_step_count": baseline_context["baseline_prefix_step_count"],
        "baseline_step_count": baseline_context["baseline_step_count"],
        "baseline_model": baseline_context["baseline_model"],
    }


# ── Parsing model output ───────────────────────────────────────────────────────

def parse_steps(text: str, k: int) -> list[str] | None:
    """
    Extract exactly k steps from model output.
    Strips 'Step N:' prefixes — step numbers are assigned by Script 3
    at trace-build time based on the baseline CoT midpoint.
    """
    import re

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    steps: list[str] = []
    for line in lines:
        cleaned = re.sub(r"^(Step\s+\d+\s*:?\s*|\d+[\.\)]\s*)", "", line).strip()
        if cleaned:
            steps.append(cleaned)

    if len(steps) == k:
        return steps
    if len(steps) > k:
        return steps[:k]
    return None


# ── API calls ──────────────────────────────────────────────────────────────────

def call_anthropic(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float = 0.8,
    max_retries: int = 3,
    system: str | None = None,
) -> str:
    try:
        import anthropic
    except ImportError:
        print("Install anthropic: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            response = client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  API error (attempt {attempt + 1}): {exc} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")


def call_openai(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float = 0.8,
    max_retries: int = 3,
    system: str | None = None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  API error (attempt {attempt + 1}): {exc} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")


def call_llm(
    prompt: str,
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float = 0.8,
    system: str | None = None,
) -> str:
    if provider == "anthropic":
        return call_anthropic(prompt, model, max_tokens, temperature, system=system)
    if provider == "openai":
        return call_openai(prompt, model, max_tokens, temperature, system=system)
    raise ValueError(f"Unknown provider: {provider}")


def resolve_provider(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "anthropic"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)

    parser = argparse.ArgumentParser(description="Generate LLM injections (Script 2).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / cfg.get("tasks", "data/experiment_tasks.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / cfg.get("output", "data/injections_generated.jsonl"),
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=ROOT / cfg.get("baseline_file", "data/baseline_cots.jsonl"),
        help="Verified baseline CoTs used to provide midpoint prefix context when available",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=cfg.get("baseline_model"),
        help=(
            "Generate injections for this experiment model only (uses its baselines; "
            "skips target_models from config)"
        ),
    )
    parser.add_argument(
        "--target-models",
        type=str,
        nargs="+",
        default=None,
        help="Override target_models from config (space-separated experiment model names)",
    )
    parser.add_argument(
        "--no-baseline-context",
        action="store_true",
        help="Fallback mode: generate prompts without baseline CoT prefix context",
    )
    parser.add_argument("--domain", type=str, default=None, help="Filter domain (math/science/logic)")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to process (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only, no API calls")
    parser.add_argument(
        "--provider",
        type=str,
        default=cfg.get("provider", None),
        choices=["anthropic", "openai"],
        help="LLM provider (default: anthropic if ANTHROPIC_API_KEY set)",
    )
    parser.add_argument("--model", type=str, default=cfg.get("model", DEFAULT_MODEL))
    parser.add_argument("--k-values", type=int, nargs="+", default=cfg.get("k_values", [1, 3, 5]))
    parser.add_argument(
        "--error-types",
        type=str,
        nargs="+",
        default=cfg.get("error_types", ["incorrect_reasoning", "paraphrased_shift", "off_topic"]),
    )
    parser.add_argument(
        "--length-levels",
        type=str,
        nargs="+",
        default=list(cfg.get("length_levels", LENGTH_INSTRUCTIONS).keys())
        if isinstance(cfg.get("length_levels"), dict)
        else ["short", "long"],
    )
    parser.add_argument(
        "--injection-strategy",
        type=str,
        default=cfg.get("injection_strategy", "persistent_agent"),
        choices=["persistent_agent", "independent"],
        help=(
            "persistent_agent (default): stateful Agent B, k sequential calls each "
            "conditioned on the role prompt, baseline prefix, and prior steps. "
            "independent: legacy single-call mode emitting all k steps at once."
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    apply_config_overrides(cfg)
    use_persistent = args.injection_strategy == "persistent_agent"
    strategy_label = (
        INJECTION_STRATEGY_PERSISTENT if use_persistent else INJECTION_STRATEGY_INDEPENDENT
    )

    provider = resolve_provider(args.provider)
    temperature = cfg.get("temperature", 0.8)
    unsupported_error_types = sorted(set(args.error_types) - set(ERROR_TYPE_INSTRUCTIONS))
    if unsupported_error_types:
        print(
            "Unsupported error type(s): "
            + ", ".join(unsupported_error_types)
            + ". Supported types are exactly: "
            + ", ".join(ERROR_TYPE_INSTRUCTIONS),
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.dry_run:
        key_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        if key_var not in os.environ:
            print(f"Set {key_var} before running without --dry-run", file=sys.stderr)
            sys.exit(1)

    # Load tasks
    tasks = []
    with args.tasks.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                if args.domain is None or t.get("domain") == args.domain:
                    tasks.append(t)
    if args.limit:
        tasks = tasks[: args.limit]

    target_models = resolve_target_models(
        cfg,
        baseline_model=args.baseline_model,
        cli_models=args.target_models,
    )
    if not target_models:
        print(
            "No target_models configured. Set target_models in injection_generation.yaml, "
            "model.name in experiment.yaml, or pass --target-models / --baseline-model.",
            file=sys.stderr,
        )
        sys.exit(1)

    existing = load_existing(args.output)
    combos_per_task = len(args.error_types) * len(args.k_values) * len(args.length_levels)
    total_all = len(tasks) * combos_per_task * len(target_models)
    skipped_all = 0
    for target_model in target_models:
        for task in tasks:
            for et in args.error_types:
                for k in args.k_values:
                    for ll in args.length_levels:
                        if injection_dedup_key(task["task_id"], et, k, ll, target_model) in existing:
                            skipped_all += 1

    print(f"Tasks: {len(tasks)} | Target models: {len(target_models)}")
    for name in target_models:
        print(f"  - {name}")
    print(
        f"Combinations per task per model: {combos_per_task} "
        f"({len(args.error_types)} error_types × {len(args.k_values)} k × {len(args.length_levels)} lengths)"
    )
    print(f"Total combinations: {total_all} | Already done: {skipped_all} | Remaining: {total_all - skipped_all}")
    print(f"Injection strategy: {strategy_label}"
          + (" (k sequential calls per combination)" if use_persistent else " (1 call per combination)"))
    print(f"Provider: {provider} | Generator model: {args.model} | Output: {args.output}")
    if args.no_baseline_context:
        print("Baseline prefix context: disabled (--no-baseline-context)")
    done = 0
    failed = 0

    # ── Dry run: no API calls; show deterministic, stateful placeholder steps ──
    if args.dry_run and tasks:
        dry_k = args.k_values[0]
        dry_len = args.length_levels[0]
        if use_persistent:
            print(
                f"\n--- DRY RUN (persistent_agent) — {dry_k} stateful steps per "
                f"error type (length={dry_len}) ---\n"
            )
        else:
            print(f"\n--- DRY RUN (independent) — one prompt per error type (k={dry_k}, {dry_len}) ---\n")
        target_model = target_models[0]
        baseline_contexts = (
            {}
            if args.no_baseline_context
            else load_baseline_contexts(args.baselines, target_model)
        )
        if not args.no_baseline_context:
            if baseline_contexts:
                print(
                    f"Baseline prefix contexts for {target_model}: "
                    f"{len(baseline_contexts)} loaded from {args.baselines}"
                )
            else:
                print(
                    f"Baseline prefix contexts for {target_model}: none at {args.baselines}; "
                    "using standalone fallback prompts"
                )
        task = tasks[0]
        baseline_context = baseline_contexts.get(task["task_id"])
        print(f"Preview target_model={target_model} task={task['task_id']}\n")
        for error_type in args.error_types:
            print(
                f"[{error_type}] task={task['task_id']} k={dry_k} length={dry_len} "
                f"prefix_used={bool(baseline_context)} strategy={strategy_label}"
            )
            print("─" * 60)
            if use_persistent:
                decoy = task.get("decoy_answer", "an incorrect value")
                role_prompt = build_role_prompt(
                    error_type, decoy, task.get("domain", "default")
                )
                print(f"AGENT ROLE (id={role_id(error_type)}):")
                print(role_prompt)
                print("·" * 60)
                prior: list[str] = []
                for i in range(dry_k):
                    step = placeholder_step(
                        task, error_type, dry_k, i, prior, baseline_context
                    )
                    prior.append(step)
                    print(f"  Step {i + 1}: {step}")
            else:
                prompt = build_prompt(
                    task, error_type, dry_k, dry_len, baseline_context=baseline_context
                )
                print(prompt)
            print("─" * 60 + "\n")
        return

    for target_model in target_models:
        baseline_contexts = (
            {}
            if args.no_baseline_context
            else load_baseline_contexts(args.baselines, target_model)
        )
        if not args.no_baseline_context:
            if baseline_contexts:
                print(
                    f"\n=== target_model={target_model} | "
                    f"{len(baseline_contexts)} baseline prefix contexts from {args.baselines} ==="
                )
            else:
                print(
                    f"\n=== target_model={target_model} | no baselines at {args.baselines}; "
                    "standalone fallback prompts ==="
                )

        work = [
            (task, et, k, ll)
            for task in tasks
            for et in args.error_types
            for k in args.k_values
            for ll in args.length_levels
        ]
        model_skipped = sum(
            1
            for task, et, k, ll in work
            if injection_dedup_key(task["task_id"], et, k, ll, target_model) in existing
        )
        model_remaining = len(work) - model_skipped
        model_done = 0

        for task, error_type, k, length_level in work:
            key = injection_dedup_key(
                task["task_id"], error_type, k, length_level, target_model
            )
            if key in existing:
                continue

            baseline_context = baseline_contexts.get(task["task_id"])
            max_tokens = MAX_TOKENS[length_level]

            try:
                if use_persistent:
                    def _step_gen(system_prompt: str, user_prompt: str) -> str:
                        raw = call_llm(
                            user_prompt, provider, args.model, max_tokens, temperature,
                            system=system_prompt,
                        )
                        time.sleep(0.3)
                        return parse_single_step(raw)

                    steps, role_prompt = generate_persistent_steps(
                        task, error_type, k, length_level, baseline_context, _step_gen
                    )
                    if any(not s for s in steps) or len(steps) != k:
                        print(
                            f"  [WARN] Persistent generation returned {len(steps)} "
                            f"non-empty steps for {key} — skipped (will retry on resume)"
                        )
                        failed += 1
                        continue
                    agent_role_value = role_prompt
                    agent_role_id_value = role_id(error_type)
                else:
                    prompt = build_prompt(
                        task, error_type, k, length_level, baseline_context=baseline_context
                    )
                    raw = call_llm(prompt, provider, args.model, max_tokens, temperature)
                    steps = parse_steps(raw, k)
                    if steps is None or any(not s for s in steps) or len(steps) != k:
                        preview = raw[:100] if steps is None else repr(steps)
                        print(
                            f"  [WARN] Could not parse {k} valid steps for {key}. "
                            f"Raw/parsed: {preview!r} — skipped (will retry on resume)"
                        )
                        failed += 1
                        continue
                    agent_role_value = None
                    agent_role_id_value = "independent"
                    time.sleep(0.3)

                record = {
                    "task_id": task["task_id"],
                    "domain": task.get("domain"),
                    "error_type": error_type,
                    "k": k,
                    "length_level": length_level,
                    "target_model": target_model,
                    "injection_steps": steps,
                    "generated_by": f"{provider}:{args.model}",
                    "injection_strategy": strategy_label,
                    "agent_role_id": agent_role_id_value,
                    "agent_role": agent_role_value,
                    "decoy_used": str(task.get("decoy_answer", ""))
                    if error_type != "paraphrased_shift"
                    else "",
                    **injection_metadata(baseline_context),
                }
                append_record(args.output, record)
                existing.add(key)
                done += 1
                model_done += 1

                print(
                    f"[{model_done:4d}/{model_remaining}] {target_model.split('/')[-1][:24]:24s} | "
                    f"{task['task_id']:20s} | {error_type:20s} | k={k} {length_level:5s} | ok"
                )

            except Exception as exc:
                print(f"  [ERROR] {key}: {exc}")
                failed += 1

    print(f"\nDone. Generated: {done} | Failed: {failed} | Output: {args.output}")


if __name__ == "__main__":
    main()
