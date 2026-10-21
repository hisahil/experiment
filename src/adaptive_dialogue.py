"""Adaptive alternating dialogue orchestration (Experiment 1, Script 3b)."""

from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from llm_backend import (
    BaseBackend,
    GenerationResult,
    build_dialogue_messages,
    strip_think_closure,
)

Condition = Literal["adversarial", "neutral_control"]


class InjectionCircuitBreakerError(RuntimeError):
    """Raised when too many consecutive dialogues fail at the injection (Claude) API.

    The pilot died here: when Claude ran out of credit during dialogue generation,
    every dialogue went `failed` and produced 0 rows (482 lost). With
    claude_workers=8 a credit/auth outage fails a whole barrier at once, so we abort
    hard instead of looping hundreds of times.
    """


# API/outage signatures that should count toward the injection circuit breaker:
# credit/billing/auth, generic 4xx (incl. 429 rate limit), overload, and the
# "Max retries exceeded" message seen in the pilot log.
_INJECTION_API_ERROR_RE = re.compile(
    r"credit balance|insufficient.{0,20}credit|billing|quota|payment required|"
    r"authentication_error|permission_error|invalid_api_key|insufficient_quota|"
    r"rate.?limit|overloaded|max retries|\b4\d{2}\b|\b529\b",
    re.IGNORECASE,
)


def is_injection_api_error(message: str | None) -> bool:
    """True when an injection failure looks like an API/credit/outage error."""
    return bool(_INJECTION_API_ERROR_RE.search(message or ""))


@dataclass
class InjectionBreaker:
    """Tracks consecutive injection-API failures across lockstep rounds."""

    threshold: int = 5
    consecutive: int = 0

    def note_round(self, *, had_success: bool, api_failures: int) -> None:
        """Update the counter after one lockstep injection barrier.

        A successful injection in the round proves the API is alive, so the
        consecutive counter resets. Otherwise API-style failures accumulate.
        Raises InjectionCircuitBreakerError once the threshold is reached.
        """
        if had_success:
            self.consecutive = 0
            return
        if api_failures <= 0:
            return
        self.consecutive += api_failures
        if self.consecutive >= self.threshold:
            raise InjectionCircuitBreakerError(
                f"{self.consecutive} consecutive dialogues failed at the injection API "
                f"(threshold={self.threshold}). Aborting — check Anthropic credits/auth "
                "before resuming (load_existing_keys will skip completed dialogues)."
            )
TurnSource = Literal[
    "task", "baseline_prefix", "injection", "model_response", "neutral_control"
]

@dataclass
class TypedTurn:
    turn_index: int
    role: str
    source: TurnSource
    text: str
    injection_step_index: int | None = None
    error_type: str | None = None
    adaptive: bool | None = None
    decoy_used: str | None = None
    agent_role_id: str | None = None
    tokens_generated: int | None = None
    finish_reason: str | None = None
    is_intermediate: bool = False
    variant: str | None = None
    # Phase-1 instrumentation (continuation only): the input reasoning-trace token
    # count BEFORE this generation, the exact rendered-prompt hash, and the mean
    # sampled log-prob of this turn's output.
    reasoning_trace_tokens: int | None = None
    prompt_sha256: str | None = None
    mean_logprob: float | None = None
    # Empty-intermediate audit trail (Punkt 1, Schicht 2). intermediate_retried
    # marks turns whose first generation came back blank and were reseed-retried;
    # intermediate_empty marks the residual blank paragraphs that survived the
    # retry (analysis can then exclude/annotate them transparently).
    intermediate_retried: bool | None = None
    intermediate_empty: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "turn_index": self.turn_index,
            "role": self.role,
            "source": self.source,
            "text": self.text,
        }
        for key in (
            "injection_step_index",
            "error_type",
            "adaptive",
            "decoy_used",
            "agent_role_id",
            "tokens_generated",
            "finish_reason",
            "is_intermediate",
            "variant",
            "reasoning_trace_tokens",
            "prompt_sha256",
            "mean_logprob",
            "intermediate_retried",
            "intermediate_empty",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        return out


@dataclass
class DialogueState:
    """Mutable state for one adaptive dialogue run."""

    dialogue_id: str
    task: dict
    baseline: dict
    error_type: str
    k: int
    length_level: str
    seed: int
    condition: Condition
    model_name: str
    turn2_source: str
    baseline_prefix: str
    insertion_step: int
    injection_tag: str = "none"
    rounds_remaining: int = 0
    prior_injections: list[str] = field(default_factory=list)
    model_turns: list[str] = field(default_factory=list)
    dialogue_turns: list[tuple[str, str]] = field(default_factory=list)
    pending_injection: str | None = None
    typed_turns: list[TypedTurn] = field(default_factory=list)
    pregenerated_first: str | None = None
    agent_role_id: str | None = None
    failed: bool = False
    fail_reason: str | None = None
    # Continuation mode (deletion model). generation_mode="restart" keeps the
    # legacy multi-turn path untouched.
    generation_mode: str = "restart"
    accumulated_trace: str = ""
    injection_steps: list[str] = field(default_factory=list)
    trace_segments: list[dict] = field(default_factory=list)
    # Phase-1 metadata: decoy (false_consensus only) + clean-baseline step count
    # for derived metrics; start_monotonic enables dialogue_wall_time_s.
    decoy_answer: str | None = None
    baseline_step_count: int | None = None
    # Injection sub-labels propagated from Script 2. shift_subtype is only set
    # for paraphrased_shift (flip_sign / shift_target / change_number /
    # drop_constraint); off_topic_theme is only set for off_topic. Both are
    # None otherwise. Populated at state-construction time so downstream
    # records and complete_dialogues carry them (Punkt 3, Feld-Propagation).
    shift_subtype: str | None = None
    off_topic_theme: str | None = None
    start_monotonic: float | None = field(default=None, repr=False)
    _turn_counter: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.rounds_remaining <= 0:
            self.rounds_remaining = int(self.k)

    def append_turn(self, turn: TypedTurn) -> None:
        self.typed_turns.append(turn)
        self._turn_counter = max(self._turn_counter, turn.turn_index + 1)

    def next_turn_index(self) -> int:
        idx = self._turn_counter
        self._turn_counter += 1
        return idx

    def current_round(self) -> int:
        """1-based round index (1..k)."""
        return int(self.k) - int(self.rounds_remaining) + 1

    def continuation_append(self, text: str, *, separator: str = "\n\n") -> None:
        """Append text to the open-think-block trace (deletion model)."""
        if self.accumulated_trace:
            self.accumulated_trace += separator + text
        else:
            self.accumulated_trace = text

    def add_trace_segment(self, seg_type: str, text: str, **meta: Any) -> None:
        """Record one ordered trace segment for the continuation audit trail."""
        seg: dict[str, Any] = {"type": seg_type, "text": text}
        for key, val in meta.items():
            if val is not None:
                seg[key] = val
        self.trace_segments.append(seg)

    def build_messages_for_generation(self) -> list[dict[str, str]]:
        turns: list[tuple[str, str | None]] = [
            (inj, resp) for inj, resp in self.dialogue_turns
        ]
        if self.pending_injection is not None:
            turns.append((self.pending_injection, None))
        return build_dialogue_messages(
            self.task["prompt"],
            self.baseline_prefix,
            turns,
            injection_tag=self.injection_tag,  # type: ignore[arg-type]
        )

    def transcript_record(self) -> dict[str, Any]:
        labels = extract_localizer_labels(self.typed_turns)
        record = {
            "dialogue_id": self.dialogue_id,
            "task_id": self.task["task_id"],
            "model": self.model_name,
            "domain": self.task.get("domain"),
            "eval_type": self.task.get("eval_type"),
            "error_type": self.error_type if self.condition == "adversarial" else None,
            "k": self.k,
            "length_level": self.length_level,
            "seed": self.seed,
            "condition": self.condition,
            "turn2_source": self.turn2_source,
            "insertion_step": self.insertion_step,
            "generation_mode": self.generation_mode,
            "turns": [t.to_dict() for t in self.typed_turns],
            **labels,
        }
        if self.generation_mode == "continuation":
            record["accumulated_trace_len"] = len(self.accumulated_trace)
            record["prompt_render_method"] = "continuation_trace"
            record["trace_segments"] = list(self.trace_segments)
            # Full reasoning trace as an explicit field (Restart-pilot parity): the
            # shared fork-point C before stop/recovery. The two branch outputs are
            # added in build_complete_dialogue_record (stop_full_trace/recovery_full_trace).
            record["accumulated_trace"] = self.accumulated_trace
            if self.decoy_answer is not None:
                record["decoy_answer"] = self.decoy_answer
            if self.baseline_step_count is not None:
                record["baseline_step_count"] = self.baseline_step_count
        if self.shift_subtype is not None:
            record["shift_subtype"] = self.shift_subtype
        if self.off_topic_theme is not None:
            record["off_topic_theme"] = self.off_topic_theme
        return record


def extract_localizer_labels(typed_turns: list[TypedTurn]) -> dict[str, Any]:
    """Gold labels for C2 Error Localizer from typed transcript turns."""
    injected_turn_spans: list[dict[str, Any]] = []
    injected_step_offsets: list[dict[str, Any]] = []
    injection_texts: list[str] = []

    for turn in typed_turns:
        if turn.source not in ("injection", "neutral_control"):
            continue
        if turn.source == "injection":
            injection_texts.append(turn.text)
            injected_turn_spans.append({
                "turn_index": turn.turn_index,
                "injection_step_index": turn.injection_step_index,
                "text": turn.text,
                "adaptive": turn.adaptive,
                "error_type": turn.error_type,
            })
            for i, line in enumerate(_split_step_lines(turn.text)):
                injected_step_offsets.append({
                    "turn_index": turn.turn_index,
                    "injection_step_index": turn.injection_step_index,
                    "step_offset": i,
                    "text": line,
                })

    return {
        "injection_steps": injection_texts,
        "injected_turn_spans": injected_turn_spans,
        "injected_step_offsets": injected_step_offsets,
    }


def build_complete_dialogue_record(
    state: DialogueState,
    stop_res: GenerationResult,
    rec_res: GenerationResult,
) -> dict[str, Any]:
    """Full dialogue transcript including final stop and recovery assistant turns."""
    record = state.transcript_record()
    turns = list(record["turns"])
    next_idx = max((t["turn_index"] for t in turns), default=-1) + 1
    turns.append(TypedTurn(
        turn_index=next_idx,
        role="assistant",
        source="model_response",
        text=stop_res.text,
        tokens_generated=stop_res.completion_tokens,
        finish_reason=stop_res.finish_reason,
        is_intermediate=False,
        variant="stop",
        reasoning_trace_tokens=stop_res.reasoning_trace_tokens,
        prompt_sha256=stop_res.prompt_sha256,
        mean_logprob=stop_res.mean_logprob,
    ).to_dict())
    turns.append(TypedTurn(
        turn_index=next_idx + 1,
        role="assistant",
        source="model_response",
        text=rec_res.text,
        tokens_generated=rec_res.completion_tokens,
        finish_reason=rec_res.finish_reason,
        is_intermediate=False,
        variant="recovery",
        reasoning_trace_tokens=rec_res.reasoning_trace_tokens,
        prompt_sha256=rec_res.prompt_sha256,
        mean_logprob=rec_res.mean_logprob,
    ).to_dict())
    record["turns"] = turns
    record["dialogue_mode"] = (
        "continuation" if state.generation_mode == "continuation" else "adaptive_alternating"
    )

    if state.generation_mode == "continuation":
        # Two parallel branches from the shared fork-point C (accumulated_trace):
        # stop commits an answer, recovery keeps reasoning. Recovery never sees the
        # stop output, so both share the SAME input C (reasoning_trace_tokens).
        sep = "\n\n"
        c_trace = state.accumulated_trace
        record["stop_full_trace"] = (c_trace + sep + stop_res.text) if c_trace else stop_res.text
        record["recovery_full_trace"] = (c_trace + sep + rec_res.text) if c_trace else rec_res.text
        record["reasoning_trace_tokens"] = (
            stop_res.reasoning_trace_tokens
            if stop_res.reasoning_trace_tokens is not None
            else rec_res.reasoning_trace_tokens
        )
        record["stop_tokens_generated"] = stop_res.completion_tokens
        record["recovery_tokens_generated"] = rec_res.completion_tokens
        record["stop_mean_logprob"] = stop_res.mean_logprob
        record["recovery_mean_logprob"] = rec_res.mean_logprob
        # answer_top_logprobs only carried for stop (boxed-answer distribution).
        if stop_res.answer_top_logprobs is not None:
            record["stop_answer_top_logprobs"] = stop_res.answer_top_logprobs
        if state.start_monotonic is not None:
            record["dialogue_wall_time_s"] = round(time.monotonic() - state.start_monotonic, 3)
    return record


def append_jsonl_flush(path: Path | str, record: dict) -> None:
    """Append one JSONL row and flush to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        try:
            import os
            os.fsync(f.fileno())
        except OSError:
            pass


DialogueCompleteFn = Callable[[DialogueState, GenerationResult, GenerationResult], None]


def _split_step_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines if lines else [text.strip()]


def resolve_intermediate_max_tokens(
    cfg_value: int | str | None,
    baselines: list[dict],
    *,
    factor: float = 1.5,
    floor: int = 1024,
    ceil: int = 4096,
) -> int:
    """Derive intermediate cap from baseline segment lengths (words as proxy)."""
    if isinstance(cfg_value, int) and cfg_value > 0:
        return cfg_value
    if cfg_value not in (None, "auto"):
        try:
            return max(1, int(cfg_value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass

    segment_words: list[int] = []
    for rec in baselines:
        steps = list(rec.get("steps") or [])
        if not steps:
            continue
        midpoint = len(steps) // 2
        for step in steps[:midpoint]:
            segment_words.append(len(str(step).split()))
    if not segment_words:
        return floor
    mean_words = sum(segment_words) / len(segment_words)
    # Rough tokens ≈ 1.3 * words for English reasoning text
    estimated = int(round(mean_words * 1.3 * factor))
    return max(floor, min(ceil, estimated))


def init_dialogue_state(
    *,
    task: dict,
    baseline: dict,
    baseline_prefix: str,
    insertion_step: int,
    error_type: str,
    k: int,
    length_level: str,
    seed: int,
    condition: Condition,
    model_name: str,
    turn2_source: str,
    pregenerated_first: str | None,
    agent_role_id: str | None,
    injection_tag: str = "none",
) -> DialogueState:
    dialogue_id = str(uuid.uuid4())
    state = DialogueState(
        dialogue_id=dialogue_id,
        task=task,
        baseline=baseline,
        error_type=error_type,
        k=int(k),
        length_level=length_level,
        seed=seed,
        condition=condition,
        model_name=model_name,
        turn2_source=turn2_source,
        baseline_prefix=baseline_prefix,
        insertion_step=insertion_step,
        injection_tag=injection_tag,
        pregenerated_first=pregenerated_first,
        agent_role_id=agent_role_id,
    )
    state.append_turn(TypedTurn(
        turn_index=state.next_turn_index(),
        role="user",
        source="task",
        text=task["prompt"],
    ))
    state.append_turn(TypedTurn(
        turn_index=state.next_turn_index(),
        role="assistant",
        source="baseline_prefix",
        text=baseline_prefix,
    ))
    return state


InjectionFn = Callable[[DialogueState], str]


def _attach_injection_turn(
    state: DialogueState,
    injection_text: str,
    *,
    step_index: int,
    adaptive: bool,
    decoy_used: str | None,
) -> None:
    source: TurnSource = (
        "neutral_control" if state.condition == "neutral_control" else "injection"
    )
    state.prior_injections.append(injection_text)
    state.pending_injection = injection_text
    state.append_turn(TypedTurn(
        turn_index=state.next_turn_index(),
        role="user",
        source=source,
        text=injection_text,
        injection_step_index=step_index,
        error_type=state.error_type if state.condition == "adversarial" else None,
        adaptive=adaptive,
        decoy_used=decoy_used,
        agent_role_id=state.agent_role_id,
    ))


def _attach_model_turn(
    state: DialogueState,
    text: str,
    *,
    tokens: int,
    finish_reason: str | None,
    is_intermediate: bool,
) -> None:
    state.model_turns.append(text)
    if state.pending_injection is not None:
        state.dialogue_turns.append((state.pending_injection, text))
        state.pending_injection = None
    state.append_turn(TypedTurn(
        turn_index=state.next_turn_index(),
        role="assistant",
        source="model_response",
        text=text,
        tokens_generated=tokens,
        finish_reason=finish_reason,
        is_intermediate=is_intermediate,
    ))


def run_lockstep_round(
    active: list[DialogueState],
    backend: BaseBackend,
    get_injection: InjectionFn,
    *,
    adaptive_from: int,
    intermediate_max_tokens: int,
    stop_max_tokens: int,
    recovery_max_tokens: int,
    temperature: float,
    top_p: float,
    select_stop_prefill: Callable[[str, str | None], str],
    claude_workers: int = 8,
    claude_retries: int = 2,
    on_dialogue_complete: DialogueCompleteFn | None = None,
    injection_breaker: InjectionBreaker | None = None,
) -> tuple[list[dict], list[DialogueState]]:
    """
    Execute one lockstep round for all active dialogues (possibly mixed k).

    Returns (final_run_records_for_completed_dialogues, still_active_states).
    """
    if not active:
        return [], []

    # Phase 1: Claude barrier — collect injections for this round
    injection_map: dict[str, str] = {}
    failed_ids: set[str] = set()

    def _inject_for_state(st: DialogueState) -> tuple[str, str | None, str | None]:
        for attempt in range(claude_retries + 1):
            try:
                text = get_injection(st)
                if text and text.strip():
                    return st.dialogue_id, text.strip(), None
            except Exception as exc:
                if attempt >= claude_retries:
                    return st.dialogue_id, None, str(exc)
        return st.dialogue_id, None, "empty injection"

    api_failures = 0
    with ThreadPoolExecutor(max_workers=claude_workers) as pool:
        futures = {pool.submit(_inject_for_state, st): st for st in active}
        for fut in as_completed(futures):
            st = futures[fut]
            did, text, err = fut.result()
            if text is None:
                st.failed = True
                st.fail_reason = err
                failed_ids.add(did)
                if is_injection_api_error(err):
                    api_failures += 1
            else:
                injection_map[did] = text

    # Circuit breaker: abort if injection-API failures pile up across rounds
    # (e.g. Claude credit exhaustion) instead of failing every remaining dialogue.
    if injection_breaker is not None:
        injection_breaker.note_round(
            had_success=bool(injection_map), api_failures=api_failures
        )

    still_active: list[DialogueState] = []
    intermediate_batch: list[tuple[DialogueState, list[dict[str, str]]]] = []
    final_batch: list[tuple[DialogueState, list[dict[str, str]], str]] = []

    for st in active:
        if st.dialogue_id in failed_ids:
            continue
        inj = injection_map[st.dialogue_id]
        step_index = st.current_round()
        adaptive = step_index >= adaptive_from or st.condition == "neutral_control"
        if step_index == 1 and st.pregenerated_first and st.condition == "adversarial":
            adaptive = False
        decoy = str(st.task.get("decoy_answer", "")) if st.condition == "adversarial" else None
        _attach_injection_turn(
            st,
            inj,
            step_index=step_index,
            adaptive=adaptive,
            decoy_used=decoy,
        )
        st.rounds_remaining -= 1
        messages = st.build_messages_for_generation()
        domain = st.task.get("domain", "math")
        eval_type = st.task.get("eval_type")

        if st.rounds_remaining > 0:
            intermediate_batch.append((st, messages))
        else:
            prefill = select_stop_prefill(domain, eval_type)
            final_batch.append((st, messages, prefill))
        still_active.append(st)

    # Phase 2: GPU — intermediate responses
    if intermediate_batch:
        items = [(msgs, None) for _, msgs in intermediate_batch]
        seeds = [st.seed for st, _ in intermediate_batch]
        results = backend.generate_chat_batch(
            items,
            max_new_tokens=intermediate_max_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=seeds,
        )
        for (st, _), res in zip(intermediate_batch, results):
            _attach_model_turn(
                st,
                res.text,
                tokens=res.completion_tokens,
                finish_reason=res.finish_reason,
                is_intermediate=True,
            )

    # Phase 3: GPU — final stop + recovery for dialogues finished this round
    completed_records: list[dict] = []
    if final_batch:
        stop_items = [(msgs, prefill) for _, msgs, prefill in final_batch]
        stop_seeds = [st.seed for st, _, _ in final_batch]
        stop_results = backend.generate_chat_batch(
            stop_items,
            max_new_tokens=stop_max_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=stop_seeds,
        )
        rec_items = [(msgs, None) for _, msgs, _ in final_batch]
        rec_seeds = [st.seed for st, _, _ in final_batch]
        rec_results = backend.generate_chat_batch(
            rec_items,
            max_new_tokens=recovery_max_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=rec_seeds,
        )
        for (st, _, _), stop_res, rec_res in zip(final_batch, stop_results, rec_results):
            completed_records.extend(
                _final_run_records(st, stop_res, rec_res, recovery_max_tokens)
            )
            if on_dialogue_complete is not None:
                on_dialogue_complete(st, stop_res, rec_res)
            still_active = [s for s in still_active if s.dialogue_id != st.dialogue_id]

    return completed_records, still_active


def _final_run_records(
    state: DialogueState,
    stop_res: GenerationResult,
    rec_res: GenerationResult,
    recovery_max_tokens: int,
) -> list[dict]:
    """Build stop + recovery output rows (meta filled by orchestrator)."""
    return [
        {
            "dialogue_id": state.dialogue_id,
            "variant": "stop",
            "raw_output": stop_res.text,
            "tokens_generated": stop_res.completion_tokens,
            "finish_reason": stop_res.finish_reason,
            "prompt_tokens": stop_res.prompt_tokens,
            "state": state,
        },
        {
            "dialogue_id": state.dialogue_id,
            "variant": "recovery",
            "raw_output": rec_res.text,
            "tokens_generated": rec_res.completion_tokens,
            "finish_reason": rec_res.finish_reason,
            "prompt_tokens": rec_res.prompt_tokens,
            "state": state,
            "recovery_max_tokens": recovery_max_tokens,
        },
    ]


def run_dialogue_until_complete(
    states: list[DialogueState],
    backend: BaseBackend,
    get_injection: InjectionFn,
    *,
    budget_deadline: float | None = None,
    **kwargs: Any,
) -> tuple[list[dict], list[DialogueState], list[DialogueState], str]:
    """
    Run all dialogues to completion via lockstep rounds.

    Returns (run_records, completed_states, failed_states, stopped_reason).
    stopped_reason: complete | budget_exhausted
    """
    active = [s for s in states if not s.failed]
    all_records: list[dict] = []
    max_rounds = max((s.k for s in active), default=0)
    stopped_reason = "complete"

    for _round in range(max_rounds):
        if budget_deadline is not None and time.monotonic() >= budget_deadline:
            stopped_reason = "budget_exhausted"
            break
        if not active:
            break
        round_active = [s for s in active if s.rounds_remaining > 0]
        if not round_active:
            break
        records, active = run_lockstep_round(
            round_active, backend, get_injection, **kwargs,
        )
        all_records.extend(records)

    completed = [s for s in states if not s.failed and s.rounds_remaining <= 0]
    failed = [s for s in states if s.failed]
    return all_records, completed, failed, stopped_reason


def compute_continuation_prefix(
    continuation_steps: list[str], *, separator: str = "\n\n"
) -> tuple[int, str]:
    """Midpoint prefix for the deletion model.

    Injection 1 lands after floor(n/2) steps; the rest of the baseline is deleted
    and the model generates its own paragraphs between later injections. Returns
    (midpoint, prefix) where prefix is the first floor(n/2) continuation steps
    joined by *separator* ("\\n\\n", DeepSeek's natural paragraph break).
    """
    steps = [s for s in continuation_steps if s]
    midpoint = len(steps) // 2
    prefix = separator.join(steps[:midpoint])
    return midpoint, prefix


def _continuation_round_seed(base_seed: int, round_index: int) -> int:
    """Vary the seed per round so repeated intermediate turns cannot duplicate."""
    return int(base_seed) + int(round_index)


def _normalize_trace_end(trace: str, separator: str) -> str:
    """Ensure *trace* ends with EXACTLY ONE *separator* (no doubles, no missing).

    Continuation-mode intermediate turns run with ``stop=["\\n\\n"]`` so the model
    can only emit ONE paragraph per round. If the trace already ends directly with
    the injection text (no trailing newline), the model's very first token is a
    ``\\n\\n``, which fires the stop immediately and produces an empty segment.
    Terminating the prompt with exactly one separator makes the first generated
    token content, so the stop fires at the paragraph END. Idempotent: repeated
    trailing separators are collapsed to one.
    """
    if not trace or not separator:
        return trace
    stripped = trace.rstrip("\n")
    return stripped + separator


def run_continuation_dialogues(
    states: list[DialogueState],
    backend: BaseBackend,
    *,
    intermediate_max_tokens: int,
    stop_max_tokens: int,
    recovery_max_tokens: int,
    temperature: float,
    top_p: float,
    select_stop_prefill: Callable[[str, str | None], str],
    trace_separator: str = "\n\n",
    intermediate_stop: str = "\n\n",
    system: str | None = None,
    recovery_system: str | None = None,
    on_dialogue_complete: DialogueCompleteFn | None = None,
    budget_deadline: float | None = None,
    logprobs: int | None = None,
) -> tuple[list[dict], list[DialogueState], list[DialogueState], str]:
    """Run continuation-mode (deletion model) dialogues to completion.

    Per round r (1..k) for every still-active dialogue:
      1. append pre-generated injection_steps[r-1] to accumulated_trace
      2. if r < k: generate ONE paragraph (stop at "\\n\\n"), strip any </think>
         closure, append it (model's own intermediate reasoning M_r)
      3. if r == k: generate final stop (with closing prefill) and recovery
         (no prefill) from the SAME trace.

    All injections are pre-generated (no live Claude). Dialogues are batched per
    round at the GPU level. Returns (records, completed, failed, stopped_reason).
    """
    active = [s for s in states if not s.failed]
    start = time.monotonic()
    for st in active:
        st.generation_mode = "continuation"
        # dialogue_wall_time_s = elapsed from batch start to this dialogue's
        # completion (longer for higher k). All dialogues start together here.
        if st.start_monotonic is None:
            st.start_monotonic = start
    all_records: list[dict] = []
    max_rounds = max((s.k for s in active), default=0)
    remaining = list(active)
    stopped_reason = "complete"

    for r in range(1, max_rounds + 1):
        if budget_deadline is not None and time.monotonic() >= budget_deadline:
            stopped_reason = "budget_exhausted"
            break
        round_states = [s for s in remaining if not s.failed and s.k >= r]
        if not round_states:
            continue

        intermediate: list[DialogueState] = []
        final: list[DialogueState] = []
        for st in round_states:
            inj = st.injection_steps[r - 1] if (r - 1) < len(st.injection_steps) else ""
            if not inj or not inj.strip():
                st.failed = True
                st.fail_reason = f"missing pre-generated injection step {r}"
                continue
            st.continuation_append(inj.strip(), separator=trace_separator)
            adversarial = st.condition == "adversarial"
            st.add_trace_segment(
                "injection",
                inj.strip(),
                step_index=r,
                error_type=st.error_type if adversarial else None,
            )
            st.append_turn(TypedTurn(
                turn_index=st.next_turn_index(),
                role="user",
                source=("injection" if adversarial else "neutral_control"),
                text=inj.strip(),
                injection_step_index=r,
                error_type=st.error_type if adversarial else None,
                adaptive=False,
                agent_role_id=st.agent_role_id,
            ))
            st.rounds_remaining -= 1
            (intermediate if r < st.k else final).append(st)

        # Intermediate: one paragraph per dialogue, keep the think block open.
        #
        # Schicht 1 (structural fix for empty intermediate segments): normalise
        # each trace to end with EXACTLY ONE trace_separator BEFORE generation,
        # so the model's first token is content, not a leading "\n\n" that would
        # trip stop=["\n\n"] immediately. Then splice the paragraph in directly
        # (no auto-separator) — the separator already lives at the trace end.
        #
        # Schicht 2 (guarantee + audit): after the batch, detect empty segments
        # (text.strip()=="" or tokens<=1) and reseed-retry ONLY those cases via a
        # small second batch with seed+offset, BEFORE the next round starts (so
        # the trace stays coherent for subsequent injections). If a retry still
        # comes back empty, flag the segment as intermediate_empty=True instead
        # of forcing content — that keeps the analysis transparent.
        if intermediate:
            for st in intermediate:
                st.accumulated_trace = _normalize_trace_end(
                    st.accumulated_trace, trace_separator
                )
            items = [(st.task["prompt"], st.accumulated_trace, None) for st in intermediate]
            seeds = [_continuation_round_seed(st.seed, r) for st in intermediate]
            results = backend.generate_from_trace_batch(
                items,
                max_new_tokens=intermediate_max_tokens,
                temperature=temperature,
                top_p=top_p,
                seeds=seeds,
                stop=[intermediate_stop] if intermediate_stop else None,
                system=system,
                logprobs=logprobs,
            )

            # Schicht 2: gather blank results and reseed-retry them once.
            retried_indices: set[int] = set()
            blank_indices = [
                idx for idx, res in enumerate(results)
                if not strip_think_closure(res.text).strip()
            ]
            if blank_indices:
                retry_states = [intermediate[i] for i in blank_indices]
                retry_items = [
                    (st.task["prompt"], st.accumulated_trace, None) for st in retry_states
                ]
                # Distinct-seed retry so sampled decoding cannot reproduce the
                # exact same empty output; the "+ 1013" offset is a coprime
                # nudge, small enough to stay in the model's usual seed range.
                retry_seeds = [
                    _continuation_round_seed(st.seed, r) + 1013 for st in retry_states
                ]
                retry_results = backend.generate_from_trace_batch(
                    retry_items,
                    max_new_tokens=intermediate_max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seeds=retry_seeds,
                    stop=[intermediate_stop] if intermediate_stop else None,
                    system=system,
                    logprobs=logprobs,
                )
                for idx, retry_res in zip(blank_indices, retry_results):
                    results[idx] = retry_res
                    retried_indices.add(idx)

            for idx, (st, res) in enumerate(zip(intermediate, results)):
                paragraph = strip_think_closure(res.text).strip()
                is_empty = not paragraph
                is_retried = idx in retried_indices
                # Trace already ends with one separator (Schicht 1); append the
                # paragraph directly. Empty paragraphs leave the trace unchanged.
                if paragraph:
                    st.accumulated_trace = st.accumulated_trace + paragraph
                seg_meta: dict[str, Any] = dict(
                    step_index=r,
                    finish_reason=res.finish_reason,
                    tokens=res.completion_tokens,
                    reasoning_trace_tokens=res.reasoning_trace_tokens,
                    prompt_sha256=res.prompt_sha256,
                    mean_logprob=res.mean_logprob,
                )
                if is_retried:
                    seg_meta["intermediate_retried"] = True
                if is_empty:
                    seg_meta["intermediate_empty"] = True
                st.add_trace_segment("model", paragraph, **seg_meta)
                st.model_turns.append(paragraph)
                turn = TypedTurn(
                    turn_index=st.next_turn_index(),
                    role="assistant",
                    source="model_response",
                    text=paragraph,
                    tokens_generated=res.completion_tokens,
                    finish_reason=res.finish_reason,
                    is_intermediate=True,
                    reasoning_trace_tokens=res.reasoning_trace_tokens,
                    prompt_sha256=res.prompt_sha256,
                    mean_logprob=res.mean_logprob,
                )
                if is_retried:
                    turn.intermediate_retried = True
                if is_empty:
                    turn.intermediate_empty = True
                st.append_turn(turn)

        # Final: stop (closing prefill) + recovery (no prefill), SAME trace.
        if final:
            domains = [(st.task.get("domain", "math"), st.task.get("eval_type")) for st in final]
            stop_items = [
                (st.task["prompt"], st.accumulated_trace, select_stop_prefill(dom, et))
                for st, (dom, et) in zip(final, domains)
            ]
            stop_seeds = [st.seed for st in final]
            stop_results = backend.generate_from_trace_batch(
                stop_items,
                max_new_tokens=stop_max_tokens,
                temperature=temperature,
                top_p=top_p,
                seeds=stop_seeds,
                system=system,
                logprobs=logprobs,
            )
            rec_items = [(st.task["prompt"], st.accumulated_trace, None) for st in final]
            rec_seeds = [st.seed for st in final]
            # Recovery has no prefill; recovery_system adds the math \boxed{} format
            # instruction so the parser sees a standard answer (continuation runs are
            # math-only). Stop/intermediate keep the base system.
            rec_results = backend.generate_from_trace_batch(
                rec_items,
                max_new_tokens=recovery_max_tokens,
                temperature=temperature,
                top_p=top_p,
                seeds=rec_seeds,
                system=recovery_system or system,
                logprobs=logprobs,
            )
            for st, stop_res, rec_res in zip(final, stop_results, rec_results):
                st.add_trace_segment(
                    "stop", stop_res.text, variant="stop",
                    tokens=stop_res.completion_tokens,
                    finish_reason=stop_res.finish_reason,
                    reasoning_trace_tokens=stop_res.reasoning_trace_tokens,
                    prompt_sha256=stop_res.prompt_sha256,
                    mean_logprob=stop_res.mean_logprob,
                )
                st.add_trace_segment(
                    "recovery", rec_res.text, variant="recovery",
                    tokens=rec_res.completion_tokens,
                    finish_reason=rec_res.finish_reason,
                    reasoning_trace_tokens=rec_res.reasoning_trace_tokens,
                    prompt_sha256=rec_res.prompt_sha256,
                    mean_logprob=rec_res.mean_logprob,
                )
                all_records.extend(
                    _final_run_records(st, stop_res, rec_res, recovery_max_tokens)
                )
                if on_dialogue_complete is not None:
                    on_dialogue_complete(st, stop_res, rec_res)
            done_ids = {st.dialogue_id for st in final}
            remaining = [s for s in remaining if s.dialogue_id not in done_ids]

    completed = [s for s in states if not s.failed and s.rounds_remaining <= 0]
    failed = [s for s in states if s.failed]
    return all_records, completed, failed, stopped_reason


def append_transcript_jsonl(path, state: DialogueState) -> None:
    append_jsonl_flush(path, state.transcript_record())
