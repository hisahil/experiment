"""LLM backends for local dry-run and GPU inference."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    wall_time_s: float
    # Set by generate_chat when the backend can distinguish EOS vs length cap.
    finish_reason: str | None = None  # "stop" | "length" | None
    # --- Phase-1 instrumentation (continuation runs). All optional; only the
    # vLLM / continuation path fills them. Dry-run / Transformers leave them None
    # (or use a word-count fallback for the trace token count) so legacy callers
    # are unaffected. ---
    # Prompt audit: exact string sent to the backend (after chat-template render).
    prompt_sha256: str | None = None
    rendered_prompt: str | None = None  # only carried for audit-sample persistence
    # Token count of the continuation reasoning trace (accumulated_trace) at the
    # moment of THIS call — i.e. the shared fork-point C for stop/recovery, or the
    # growing context for intermediate turns. Excludes chat template / task / system.
    reasoning_trace_tokens: int | None = None
    # Confidence: mean log-probability over generated tokens (perplexity proxy).
    mean_logprob: float | None = None
    # Top log-probs of the FIRST generated token (for stop: distribution over the
    # boxed answer; derived_metrics compares decoy vs gold for false_consensus).
    answer_top_logprobs: list[dict] | None = None
    # Batch timing (attached per-result by generate_from_trace_batch). Per-sample
    # values are explicitly approximate: vLLM batches in parallel, so elapsed/n is
    # NOT a true per-sample latency. Use batch_wall_time_s + tokens_per_second_batch
    # for cost/ETA; surface per-sample numbers only as approx_*.
    batch_wall_time_s: float | None = None
    batch_size: int | None = None
    approx_wall_time_s: float | None = None
    tokens_per_second_batch: float | None = None
    tokens_per_second_sample_approx: float | None = None


def _sha256(text: str) -> str:
    """Stable hash of the exact rendered prompt (continuation-mode audit)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce_prompt_string(prompt: Any, tokenizer: Any = None) -> str:
    """Normalize a prompt to a string for hashing, audit logs, and vLLM input.

    Mistral/Ministral tokenizers may return a message list (or token-id list)
    from ``apply_chat_template`` even when ``tokenize=False`` is requested.
    Downstream code (_sha256, rendered_prompt, vLLM batch) expects str.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        if prompt and isinstance(prompt[0], int):
            if tokenizer is not None and hasattr(tokenizer, "decode"):
                try:
                    return tokenizer.decode(prompt)
                except Exception:
                    pass
        if prompt and isinstance(prompt[0], dict):
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                try:
                    rendered = tokenizer.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True,
                    )
                    if isinstance(rendered, str):
                        return rendered
                except Exception:
                    pass
            return _manual_chat_prompt(prompt)
    return str(prompt)


def count_tokens(tokenizer: Any, text: str) -> int:
    """Token count via the model tokenizer; word-count fallback when unavailable.

    Used for reasoning_trace_tokens and offline baseline tokenisation. The
    fallback (whitespace split) is only hit on the dry-run backend; the real
    vLLM/Transformers backends always carry a tokenizer.
    """
    if not text:
        return 0
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return len(text.split())


def _extract_vllm_logprobs(
    completion: Any,
) -> tuple[float | None, list[dict] | None]:
    """Derive (mean_logprob, answer_top_logprobs) from a vLLM CompletionOutput.

    mean_logprob: arithmetic mean of the SAMPLED token log-probs (a perplexity
    proxy; lower = less confident). answer_top_logprobs: the top-k candidate
    log-probs at the FIRST generated position — for stop runs this is the
    distribution over the boxed answer token, so derived_metrics can compare the
    decoy vs gold mass for false_consensus.

    Defensive across vLLM versions: returns (None, None) when logprobs were not
    requested or the structure is unexpected.
    """
    token_logprobs = getattr(completion, "logprobs", None)
    if not token_logprobs:
        return None, None
    token_ids = list(getattr(completion, "token_ids", None) or [])

    def _lp_value(entry: Any) -> float | None:
        # vLLM stores a Logprob dataclass (.logprob) keyed by token id.
        lp = getattr(entry, "logprob", None)
        return float(lp) if lp is not None else None

    # Mean over the sampled token at each step.
    sampled: list[float] = []
    for i, step in enumerate(token_logprobs):
        if not isinstance(step, dict) or not step:
            continue
        chosen = None
        if i < len(token_ids) and token_ids[i] in step:
            chosen = _lp_value(step[token_ids[i]])
        if chosen is None:
            # Fall back to the highest-prob entry when the sampled id is absent.
            vals = [v for v in (_lp_value(e) for e in step.values()) if v is not None]
            chosen = max(vals) if vals else None
        if chosen is not None:
            sampled.append(chosen)
    mean_lp = (sum(sampled) / len(sampled)) if sampled else None

    # Top candidates at the first generated position.
    top: list[dict] | None = None
    first = token_logprobs[0] if token_logprobs else None
    if isinstance(first, dict) and first:
        items: list[dict] = []
        for tid, entry in first.items():
            lp = _lp_value(entry)
            if lp is None:
                continue
            items.append({
                "token_id": int(tid),
                "token": getattr(entry, "decoded_token", None),
                "logprob": lp,
            })
        items.sort(key=lambda d: d["logprob"], reverse=True)
        top = items or None
    return mean_lp, top


Message = dict[str, str]


def _expand_seeds(seeds: list[int] | int | None, n: int) -> list[int | None]:
    """Normalise a seed argument to one value per prompt.

    None -> all None (non-deterministic), int -> same seed for every prompt,
    list -> used as-is (must match n). Keeps per-prompt seeds aligned with the
    batch so reproducibility survives vLLM batching.
    """
    if seeds is None:
        return [None] * n
    if isinstance(seeds, int):
        return [seeds] * n
    if len(seeds) != n:
        raise ValueError(f"seeds length {len(seeds)} != number of prompts {n}")
    return list(seeds)


COLLABORATING_AGENT_TAG = "[Note from collaborating agent]:"

InjectionTagMode = str  # "none" | "collaborating_agent"


def _strip_think_tags(text: str) -> str:
    """Remove accidental think tags so DeepSeek template stripping cannot delete content."""
    return re.sub(r"</?think>", "", text)


def format_injection_text(
    injection_text: str,
    *,
    injection_tag: InjectionTagMode = "none",
) -> str:
    """Format a single user-turn injection (plain text by default, no tag)."""
    if injection_tag == "collaborating_agent":
        return f"{COLLABORATING_AGENT_TAG}\n{injection_text}"
    if injection_tag != "none":
        raise ValueError(f"Unknown injection_tag: {injection_tag!r}")
    return injection_text


def format_collaborating_note(injection_steps: list[str]) -> str:
    body = "\n".join(injection_steps)
    return f"{COLLABORATING_AGENT_TAG}\n{body}"


DialogueTurn = tuple[str, str | None]
"""(user_injection_text, model_response_or_none). None on the final injection turn."""


def build_dialogue_messages(
    problem: str,
    turn2_content: str,
    turns: list[DialogueTurn],
    *,
    injection_tag: InjectionTagMode = "none",
) -> list[Message]:
    """
    Build an alternating multi-turn dialog for adaptive injection runs.

    Turn 1: user = problem
    Turn 2: assistant = turn2_content (typically baseline prefix)
    Then for each (injection, model_response) in turns:
      user = injection (optionally tagged)
      assistant = model_response (skipped when None — final injection turn)

    The last turn in ``turns`` must have model_response=None (model generates next).
    """
    if not turns:
        raise ValueError("turns must be non-empty")

    messages: list[Message] = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": _strip_think_tags(turn2_content)},
    ]

    for i, (injection_text, model_response) in enumerate(turns):
        is_final = i == len(turns) - 1
        if is_final and model_response is not None:
            raise ValueError("Final turn must have model_response=None")
        if not is_final and model_response is None:
            raise ValueError(f"Non-final turn {i + 1} requires model_response")

        messages.append({
            "role": "user",
            "content": format_injection_text(injection_text, injection_tag=injection_tag),
        })
        if model_response is not None:
            messages.append({
                "role": "assistant",
                "content": _strip_think_tags(model_response),
            })

    return messages


def build_perturbed_messages(
    problem: str,
    baseline_prefix: str,
    injection_steps: list[str],
    *,
    reasoning_model: bool = False,
    trace_in_user_turn: bool = False,
    injection_tag: InjectionTagMode = "collaborating_agent",
) -> list[Message]:
    """
    Build the multi-turn dialog for a perturbed run.

    Multi-agent mapping: the baseline prefix is the receiving model's OWN prior
    reasoning, the collaborating-agent note is the EXTERNAL agent's contribution,
    and the model then continues (recovery) or is asked to stop.

    The prefix is placed as plain visible text and is never wrapped in
    <think>...</think>. DeepSeek-R1-Distill's chat template keeps only the text
    after </think> for every assistant turn in history (it runs
    content.split('</think>')[-1]), so wrapping the prefix would delete it from
    the prompt entirely. DeepSeek also documents that historical reasoning is
    dropped from context on purpose, so there is no supported way to feed a hidden
    reasoning trace back on a later turn. The template opens its own <think> block
    on the generation turn via add_generation_prompt, so we leave think-handling
    to it rather than injecting tags that the template would strip.
    """
    baseline_prefix = _strip_think_tags(baseline_prefix)
    if injection_tag == "collaborating_agent":
        note = format_collaborating_note(injection_steps)
    else:
        note = format_injection_text("\n".join(injection_steps), injection_tag=injection_tag)

    # Folding everything into one user turn sidesteps assistant-history handling
    # completely, for cases where even a plain assistant-turn prefix is unwanted.
    if trace_in_user_turn:
        combined = (
            f"{problem}\n\n"
            f"Reasoning so far from a collaborating agent:\n{baseline_prefix}\n\n"
            f"{note}"
        )
        return [{"role": "user", "content": combined}]

    # Presenting the prefix as the model's own prior visible turn keeps the
    # external/internal split intact and survives every chat template we use.
    return [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": baseline_prefix},
        {"role": "user", "content": note},
    ]


def _manual_chat_prompt(messages: list[Message]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"{role}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def messages_to_prompt(
    tokenizer: Any,
    messages: list[Message],
    *,
    prefill: str | None = None,
) -> str:
    """Render chat messages to a single prompt string, optionally continuing assistant text.

    Chat-template prefill strategy: when *prefill* is supplied we append it as
    a partial assistant message **before** calling ``apply_chat_template`` and
    set ``add_generation_prompt=False``.  This ensures the prefill lands inside
    the assistant turn delimiters (e.g. DeepSeek-R1's ``<|assistant|>`` block)
    rather than being naively concatenated after the generation-prompt marker,
    which would place it outside the expected turn for many chat templates.
    """
    prompt: str | None = None

    # When prefill is provided, inject it as a partial assistant message so the
    # chat template wraps it correctly inside the assistant turn.
    if prefill:
        template_messages = list(messages) + [{"role": "assistant", "content": prefill}]
        add_gen_prompt = False
    else:
        template_messages = messages
        add_gen_prompt = True

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            if prefill:
                # Preferred: continue_final_message avoids spurious EOT tokens inside the prefill
                # (e.g. <|eot_id|> after the prefill text in Llama-3 templates).
                try:
                    prompt = tokenizer.apply_chat_template(
                        template_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        continue_final_message=True,
                    )
                except TypeError:
                    # Older tokenizer versions lack continue_final_message — fall back.
                    prompt = tokenizer.apply_chat_template(
                        template_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
            else:
                prompt = tokenizer.apply_chat_template(
                    template_messages,
                    tokenize=False,
                    add_generation_prompt=add_gen_prompt,
                )
        except (TypeError, ValueError, NotImplementedError):
            try:
                prompt = tokenizer.apply_chat_template(template_messages, tokenize=False)
                if not prefill and not prompt.rstrip().endswith(("assistant", "Assistant")):
                    prompt = prompt.rstrip() + "\nassistant\n"
            except (TypeError, ValueError, NotImplementedError):
                prompt = None

    if prompt is None:
        prompt = _manual_chat_prompt(messages)
        if prefill:
            prompt = f"{prompt}{prefill}"

    return prompt


def _infer_prompt_render_method(tokenizer: Any) -> str:
    """chat_template when a tokenizer can apply it; otherwise manual fallback."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return "chat_template"
    return "manual"


# ── Continuation mode (Recum-style open think block) ──────────────────────────
#
# Unlike the restart path (build_dialogue_messages + generate_chat with closed
# assistant turns), continuation mode keeps the model's think block OPEN and
# appends the (possibly corrupted) reasoning trace raw, so the model continues
# its OWN chain of thought instead of restarting. This mirrors von Recum's
# mechanism.
#
# Cross-model portage (Phase 0): the open/close tag literals are no longer
# module constants but come from `src/model_family.py` (ThinkTags), so
# GLM-Z1 / Gemma / Ministral / Phi-4 can drop in without touching this file.
# The DeepSeek defaults below are re-exported for legacy tests that import
# THINK_CLOSE_TAG / THINK_OPEN_TAG directly; runtime code should resolve tags
# via `model_family.get_think_tags(model_name?)` instead.

THINK_CLOSE_TAG = "</think>"
THINK_OPEN_TAG = "<think>"

# Recovery has no \boxed{ prefill (unlike stop), so the model often ends with prose
# ("**Answer:** the sum is 10") that the deterministic parser must guess at. This
# system instruction standardises the math recovery output on \boxed{} to reduce
# parser false-negatives. Used only for the recovery turn of math dialogues; stop
# (prefill) and intermediate (open think block) are untouched.
MATH_RECOVERY_FORMAT_INSTRUCTION = (
    "When you finish reasoning, give exactly one final answer "
    "on its own line in the form \\boxed{...}."
)


def _resolve_think_tags(think_tags: Any) -> Any:
    """Lazy resolver: use passed-in tags, else the active model family."""
    if think_tags is not None:
        return think_tags
    # Import inside the function to avoid a circular import at module load
    # (model_family only depends on stdlib + PyYAML).
    from model_family import get_think_tags  # noqa: WPS433

    return get_think_tags()


def _manual_open_think_prompt(messages: list[Message], *, think_tags: Any = None) -> str:
    """Fallback prompt (no chat template) that ends inside an open think block."""
    tags = _resolve_think_tags(think_tags)
    prompt = _manual_chat_prompt(messages)
    return f"{prompt}\n{tags.open}"


def build_continuation_prompt(
    tokenizer: Any,
    problem: str,
    reasoning_trace: str,
    *,
    system: str | None = None,
    stop_prefill: str | None = None,
    think_tags: Any = None,
    tokenizer_extra_kwargs: dict | None = None,
) -> str:
    """Render a raw-completion prompt that ends inside an OPEN assistant think block.

    Steps (Recum-style, per family):
      1. messages = [system?, user(problem)]  — no assistant turn
         - If the family requires a Thinking-Mode trigger (Gemma's `<|think|>`),
           we prepend it to the system prompt so `apply_chat_template` opens the
           thought channel instead of a bare assistant turn.
      2. apply_chat_template(add_generation_prompt=True) — opens the think block
      3. append reasoning_trace raw — the block stays open, model continues it
      4. (optional) append stop_prefill — e.g. "</think>\\n\\nTherefore, the final
         answer is \\boxed{" closes the block and forces a short committed answer.

    Cross-model:
      - `think_tags` defaults to the active model family (set via
        model_family.set_active_model_family). Fall back to DeepSeek `<think>`.
      - `tokenizer_extra_kwargs` is forwarded to `apply_chat_template` so
        Gemma can pass `enable_thinking=True`.

    The caller runs backend.generate(prompt) as raw completion (no chat history).
    """
    tags = _resolve_think_tags(think_tags)

    # tokenizer_extra_kwargs default: read from the active model family so
    # Gemma's `enable_thinking=True` is applied automatically once
    # `set_active_model_family("google/gemma-4-12B-it")` has been called
    # (typically inside `create_backend`).
    if tokenizer_extra_kwargs is None:
        try:
            from model_family import get_active_model_family  # noqa: WPS433

            extra_kwargs = dict(get_active_model_family().tokenizer_extra_kwargs)
        except Exception:
            extra_kwargs = {}
    else:
        extra_kwargs = dict(tokenizer_extra_kwargs)

    # Gemma-style trigger: inject into (or prepend to) the system message so the
    # chat template actually opens the thought channel. When no system prompt is
    # given, create a minimal one containing only the trigger.
    resolved_system: str | None = system
    if tags.trigger:
        if resolved_system:
            resolved_system = f"{tags.trigger}\n{resolved_system}"
        else:
            resolved_system = tags.trigger

    messages: list[Message] = []
    if resolved_system:
        messages.append({"role": "system", "content": resolved_system})
    messages.append({"role": "user", "content": problem})

    formatted: str | None = None
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **extra_kwargs,
            )
        except (TypeError, ValueError, NotImplementedError):
            # Some templates reject unknown kwargs (e.g. enable_thinking on
            # DeepSeek). Retry once without the extras before falling back.
            if extra_kwargs:
                try:
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except (TypeError, ValueError, NotImplementedError):
                    formatted = None
            else:
                formatted = None
    if formatted is None:
        formatted = _manual_open_think_prompt(messages, think_tags=tags)
    else:
        formatted = _coerce_prompt_string(formatted, tokenizer)

    if reasoning_trace:
        formatted += reasoning_trace
    if stop_prefill:
        formatted += stop_prefill
    return formatted


def build_baseline_prompt(
    tokenizer: Any,
    prompt_template: str,
    task_text: str,
    *,
    system_instruction: str | None = None,
    think_tags: Any = None,
    tokenizer_extra_kwargs: dict | None = None,
    use_chat_template: bool = False,
) -> str:
    """Render a Script-1 baseline prompt, optionally via the chat template.

    When ``use_chat_template`` is True (Gemma 4), builds [system?, user] messages
    and calls ``apply_chat_template(..., enable_thinking=True)`` so the model
    enters its thinking channel instead of echoing a flat-text prompt.
    """
    user_content = prompt_template.replace("{task}", task_text)
    if not use_chat_template:
        return user_content

    tags = _resolve_think_tags(think_tags)
    if tokenizer_extra_kwargs is None:
        try:
            from model_family import get_active_model_family  # noqa: WPS433

            extra_kwargs = dict(get_active_model_family().tokenizer_extra_kwargs)
        except Exception:
            extra_kwargs = {}
    else:
        extra_kwargs = dict(tokenizer_extra_kwargs)

    resolved_system = system_instruction
    if tags.trigger:
        if resolved_system:
            resolved_system = f"{tags.trigger}\n{resolved_system}"
        else:
            resolved_system = tags.trigger

    messages: list[Message] = []
    if resolved_system:
        messages.append({"role": "system", "content": resolved_system})
    messages.append({"role": "user", "content": user_content})

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **extra_kwargs,
            )
        except (TypeError, ValueError, NotImplementedError):
            if extra_kwargs:
                try:
                    return tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except (TypeError, ValueError, NotImplementedError):
                    pass
    return _manual_open_think_prompt(messages, think_tags=tags)


def prepare_baseline_raw_text(raw_text: str, *, think_tags: Any = None) -> str:
    """Strip the thinking channel and keep the visible answer region for Script 1.

    Uses the family close-tag when present; otherwise returns the raw text unchanged.
    Also recognizes malformed OLMo close markers such as ``</think#`` (missing `>`).
    """
    tags = _resolve_think_tags(think_tags)
    close = tags.close
    idx = raw_text.rfind(close)
    cut_len = len(close) if idx != -1 else -1

    # Malformed close: close-tag stem not followed by '>' (e.g. "</think#").
    # Do not treat a proper close as malformed — that would leave a stray '>'.
    if close.endswith(">"):
        stem = close[:-1]
        for m in re.finditer(re.escape(stem) + r"(?![>])[^\n]*", raw_text):
            if m.start() >= idx:
                idx = m.start()
                cut_len = m.end() - m.start()

    if idx != -1 and cut_len > 0:
        visible = raw_text[idx + cut_len :].strip()
        if visible:
            return visible
    return raw_text.strip()


def _sampling_repetition_penalty(repetition_penalty: float | None) -> float | None:
    """Return a vLLM/HF repetition_penalty when it differs from the neutral default."""
    if repetition_penalty is None:
        return None
    val = float(repetition_penalty)
    return val if val != 1.0 else None


def _truncate_at_stop(text: str, stop: list[str] | None) -> tuple[str, bool]:
    """Cut text at the earliest stop-sequence occurrence (post-hoc fallback).

    Returns (truncated_text, hit) where hit is True when a stop sequence was found.
    Used by backends without native stop-sequence support (Transformers/dry-run).
    """
    if not stop:
        return text, False
    cut = len(text)
    hit = False
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and idx < cut:
            cut = idx
            hit = True
    return (text[:cut], True) if hit else (text, False)


def strip_think_closure(text: str, *, think_tags: Any = None) -> str:
    """Keep only the text BEFORE the family close-tag for an intermediate turn.

    When the model ends an intermediate continuation with a close tag (and
    possibly a post-think answer), appending that verbatim would close the
    block and make the next generation restart. We drop the closing tag and
    everything after it so the accumulated trace stays an open think block.
    Stray opening tags are removed too.

    `think_tags` defaults to the active model family (falls back to DeepSeek
    `</think>` / `<think>`), so legacy call sites keep working without changes.
    """
    tags = _resolve_think_tags(think_tags)
    idx = text.find(tags.close)
    if idx != -1:
        text = text[:idx]
    return text.replace(tags.open, "")


class BaseBackend:
    prompt_render_method: str = "manual"
    tokenizer: Any = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seeds: list[int] | int | None = None,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> list[GenerationResult]:
        """Submit multiple prompts; base implementation is sequential."""
        seed_list = _expand_seeds(seeds, len(prompts))
        return [
            self.generate(
                p, max_new_tokens, temperature, top_p, seed=s, stop=stop,
                logprobs=logprobs, top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            for p, s in zip(prompts, seed_list)
        ]

    def generate_from_trace_batch(
        self,
        items: list[tuple[str, str, str | None]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seeds: list[int] | int | None = None,
        stop: list[str] | None = None,
        system: str | None = None,
        logprobs: int | None = None,
        think_tags: Any = None,
        tokenizer_extra_kwargs: dict | None = None,
    ) -> list[GenerationResult]:
        """Continuation-mode batch generation.

        items: list of (problem, reasoning_trace, stop_prefill). Each is rendered
        with build_continuation_prompt (open think block + raw trace) and run as a
        raw completion. When stop_prefill is set it is prepended to the result text
        (like generate_chat) so downstream answer extraction sees the full
        "...\\boxed{X}" string.

        Cross-model:
          - `think_tags` / `tokenizer_extra_kwargs` are forwarded to
            build_continuation_prompt so Gemma / Ministral / GLM prompts use
            the correct open marker and template kwargs.

        Phase-1 instrumentation attached per result (continuation only):
          - prompt_sha256 / rendered_prompt: exact string sent to the backend.
          - reasoning_trace_tokens: tokeniser count of the continuation trace
            (the accumulated_trace / fork-point C), excluding template/task/system.
          - batch_* timing: wall time measured around the whole batch call; the
            per-sample value is approximate (vLLM batches in parallel).
          - mean_logprob / answer_top_logprobs: filled when the backend supports
            logprobs (vLLM) and logprobs is requested.
        """
        # Resolve tags once so every prompt in the batch uses the same family.
        resolved_tags = _resolve_think_tags(think_tags)
        prompts = [
            _coerce_prompt_string(
                build_continuation_prompt(
                    self.tokenizer,
                    problem,
                    trace,
                    system=system,
                    stop_prefill=pref,
                    think_tags=resolved_tags,
                    tokenizer_extra_kwargs=tokenizer_extra_kwargs,
                ),
                self.tokenizer,
            )
            for problem, trace, pref in items
        ]
        trace_token_counts = [count_tokens(self.tokenizer, trace) for _, trace, _ in items]
        prompt_hashes = [_sha256(p) for p in prompts]

        t0 = time.time()
        raw_results = self.generate_batch(
            prompts, max_new_tokens, temperature, top_p, seeds=seeds, stop=stop,
            logprobs=logprobs,
        )
        batch_wall = time.time() - t0
        bsize = len(prompts)
        total_completion = sum(int(r.completion_tokens or 0) for r in raw_results)
        tps_batch = (total_completion / batch_wall) if batch_wall > 0 else None
        approx_wall = (batch_wall / bsize) if bsize else None
        tps_sample = (tps_batch / bsize) if (tps_batch is not None and bsize) else None

        results: list[GenerationResult] = []
        for (_, _, pref), prompt, phash, t_tokens, res in zip(
            items, prompts, prompt_hashes, trace_token_counts, raw_results
        ):
            text = pref + res.text if pref else res.text
            out = GenerationResult(
                text=text,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                wall_time_s=res.wall_time_s,
                finish_reason=res.finish_reason,
                prompt_sha256=phash,
                rendered_prompt=prompt,
                reasoning_trace_tokens=t_tokens,
                mean_logprob=res.mean_logprob,
                answer_top_logprobs=res.answer_top_logprobs,
                batch_wall_time_s=batch_wall,
                batch_size=bsize,
                approx_wall_time_s=approx_wall,
                tokens_per_second_batch=tps_batch,
                tokens_per_second_sample_approx=tps_sample,
            )
            results.append(out)
        return results

    def generate_chat(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        prefill: str | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        prompt = self.format_messages(messages, prefill=prefill)
        result = self.generate(prompt, max_new_tokens, temperature, top_p, seed=seed)
        if prefill:
            result = GenerationResult(
                text=prefill + result.text,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                wall_time_s=result.wall_time_s,
                finish_reason=result.finish_reason,
            )
        return result

    def generate_chat_batch(
        self,
        items: list[tuple[list[Message], str | None]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seeds: list[int] | int | None = None,
    ) -> list[GenerationResult]:
        """
        Batch variant of generate_chat.

        items: list of (messages, prefill) pairs.
        seeds: per-prompt RNG seeds (list aligned to items), a single int for
        all, or None for non-deterministic sampling.
        Returns one GenerationResult per item, in the same order.
        Prefill text is prepended to results when supplied.
        """
        prompts = [self.format_messages(msgs, prefill=pre) for msgs, pre in items]
        raw_results = self.generate_batch(prompts, max_new_tokens, temperature, top_p, seeds=seeds)
        results: list[GenerationResult] = []
        for (_, prefill), res in zip(items, raw_results):
            if prefill:
                res = GenerationResult(
                    text=prefill + res.text,
                    prompt_tokens=res.prompt_tokens,
                    completion_tokens=res.completion_tokens,
                    wall_time_s=res.wall_time_s,
                    finish_reason=res.finish_reason,
                )
            results.append(res)
        return results

    def format_messages(self, messages: list[Message], *, prefill: str | None = None) -> str:
        """Override when a backend has a native chat formatter."""
        lines: list[str] = []
        for msg in messages:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        lines.append("Assistant:")
        prompt = "\n\n".join(lines)
        if prefill:
            prompt = f"{prompt}{prefill}"
        return prompt


class DryRunBackend(BaseBackend):
    """Local backend for pipeline testing without GPU."""

    gold_hint: str | None = None

    def __init__(self) -> None:
        self.prompt_render_method = "manual"

    def format_messages(self, messages: list[Message], *, prefill: str | None = None) -> str:
        return messages_to_prompt(None, messages, prefill=prefill)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> GenerationResult:
        start = time.time()
        if "\\boxed{" in prompt:
            answer = self.gold_hint or "1"
            text = f"{answer}}}"
            elapsed = time.time() - start
            return GenerationResult(
                text=text,
                prompt_tokens=max(1, len(prompt.split())),
                completion_tokens=max(1, len(text.split())),
                wall_time_s=elapsed,
                finish_reason="stop",
            )

        if "17 plus 28" in prompt or "17+28" in prompt.replace(" ", ""):
            answer = "45"
        elif "144 divided by 12" in prompt:
            answer = "12"
        elif "x + 7 = 15" in prompt:
            answer = "8"
        elif "25 percent of 80" in prompt:
            answer = "20"
        elif "4 apples" in prompt:
            answer = "12"
        elif "chemical symbol for water" in prompt.lower():
            answer = "H2O"
        elif "Red Planet" in prompt:
            answer = "Mars"
        elif "photosynthesis" in prompt.lower():
            answer = "carbon dioxide"
        elif "unit of force" in prompt.lower():
            answer = "newton"
        elif "speed of light" in prompt.lower():
            answer = "3e8"
        elif "54" in prompt and "17 plus 28" in prompt:
            answer = "54"
        else:
            answer = "UNKNOWN"

        text = (
            "Step 1: I read the task carefully.\n"
            "Step 2: I apply the relevant reasoning.\n"
            f"Final answer: {answer}\n"
        )
        elapsed = time.time() - start
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(text.split()))
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_time_s=elapsed,
            finish_reason="stop",
        )


class TransformersBackend(BaseBackend):
    def __init__(self, model_name: str, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kwargs: dict = {"trust_remote_code": True, "device_map": "auto"}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompt_render_method = _infer_prompt_render_method(self.tokenizer)

    def format_messages(self, messages: list[Message], *, prefill: str | None = None) -> str:
        return messages_to_prompt(self.tokenizer, messages, prefill=prefill)

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seeds: list[int] | int | None = None,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> list[GenerationResult]:
        """Sequential fallback — TransformersBackend does not support true batching."""
        import warnings
        warnings.warn(
            "TransformersBackend.generate_batch() runs prompts sequentially. "
            "Use VLLMBackend for true GPU batching.",
            stacklevel=2,
        )
        seed_list = _expand_seeds(seeds, len(prompts))
        return [
            self.generate(
                p, max_new_tokens, temperature, top_p, seed=s, stop=stop, top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            for p, s in zip(prompts, seed_list)
        ]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> GenerationResult:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k is not None and top_k > 0:
                gen_kwargs["top_k"] = int(top_k)
        rep_pen = _sampling_repetition_penalty(repetition_penalty)
        if rep_pen is not None:
            gen_kwargs["repetition_penalty"] = rep_pen
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        elapsed = time.time() - start
        completion_tokens = int(generated.shape[-1])
        finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"
        # No native stop-sequence support: truncate post-hoc so continuation-mode
        # intermediate turns can be cut at the first paragraph break ("\n\n").
        text, hit = _truncate_at_stop(text, stop)
        if hit:
            finish_reason = "stop"
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_time_s=elapsed,
            finish_reason=finish_reason,
        )


class VLLMBackend(BaseBackend):
    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        *,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.90,
    ):
        from vllm import LLM

        # DeepSeek-R1-Distill defaults to a 131072-token context, which does not
        # fit the KV cache alongside the 61GB weights on an 80GB GPU. Cap
        # max_model_len (prompt + generation) so vLLM can allocate the cache.
        llm_kwargs: dict = {
            "model": model_name,
            "trust_remote_code": True,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)
        # Family-specific Python-API kwargs (Mistral needs tokenizer_mode=mistral
        # etc.; without this the HF tokenizer route mis-renders the chat
        # template and the model emits [/INST]/[SYSTEM_PROMPT] into outputs).
        try:
            from model_family import vllm_python_api_kwargs  # noqa: WPS433

            llm_kwargs.update(vllm_python_api_kwargs(model_name))
        except Exception:
            pass
        self.llm = LLM(**llm_kwargs)
        self.model_name = model_name
        self.tokenizer = None
        # Prefer the tokenizer that vLLM actually built (respects tokenizer_mode)
        # so downstream chat-template rendering matches the runtime tokenizer.
        try:
            tokenizer_getter = getattr(self.llm, "get_tokenizer", None)
            if callable(tokenizer_getter):
                self.tokenizer = tokenizer_getter()
        except Exception:
            self.tokenizer = None
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
            except Exception:
                pass
        self.prompt_render_method = _infer_prompt_render_method(self.tokenizer)
        # Cache family stop tokens so generate_batch() can inject them without a
        # yaml re-parse per call.
        try:
            from model_family import family_stop_tokens  # noqa: WPS433

            self._family_stop_tokens = list(family_stop_tokens(model_name))
        except Exception:
            self._family_stop_tokens = []

    def _merge_stop(self, caller_stop: list[str] | None) -> list[str] | None:
        """Merge caller-supplied stop sequences with family-specific stop tokens.

        Family stops (e.g. Ministral [INST]/[/INST]) prevent chat-template leaks
        where the model self-emits template tokens and simulates new user
        turns. Returns None when both sources are empty so vLLM SamplingParams
        skips the stop-list allocation entirely.
        """
        caller = list(caller_stop) if caller_stop else []
        family = list(self._family_stop_tokens or [])
        merged: list[str] = []
        for tok in caller + family:
            if tok and tok not in merged:
                merged.append(tok)
        return merged or None

    def format_messages(self, messages: list[Message], *, prefill: str | None = None) -> str:
        return messages_to_prompt(self.tokenizer, messages, prefill=prefill)

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seeds: list[int] | int | None = None,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> list[GenerationResult]:
        """Submit all prompts in one vLLM call for maximum GPU utilisation.

        seeds: per-prompt seeds become a parallel list of SamplingParams (vLLM
        accepts List[SamplingParams] aligned to the prompt list), so each prompt
        is seeded independently without splitting the batch.
        stop: optional stop sequences (e.g. ["\\n\\n"]) to cut continuation-mode
        intermediate turns after one paragraph; vLLM strips them from the output.
        logprobs: when set, request that many top log-probs per generated token so
        mean_logprob (perplexity proxy) and answer_top_logprobs (first-token
        distribution, e.g. boxed answer) can be derived.
        top_k: Card-Wert für Baseline-Sampling (Gemma=64, GLM=40). Nicht gesetzt
        im Continuation-Modus (Recum: uniform top_k=N/A).
        """
        from vllm import SamplingParams

        top_k_val = int(top_k) if (top_k is not None and top_k > 0) else -1
        merged_stop = self._merge_stop(stop)
        rep_pen = _sampling_repetition_penalty(repetition_penalty)
        seed_list = _expand_seeds(seeds, len(prompts))
        sp_kwargs: dict = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k_val,
            "stop": merged_stop,
            "logprobs": logprobs,
        }
        if rep_pen is not None:
            sp_kwargs["repetition_penalty"] = rep_pen
        if any(s is not None for s in seed_list):
            params: list[SamplingParams] | SamplingParams = [
                SamplingParams(**sp_kwargs, seed=s)
                for s in seed_list
            ]
        else:
            params = SamplingParams(**sp_kwargs)
        outputs = self.llm.generate(prompts, params)
        results: list[GenerationResult] = []
        for output in outputs:
            r = output.outputs[0]
            completion_tokens = len(r.token_ids or [])
            finish = getattr(r, "finish_reason", None) or (
                "length" if completion_tokens >= max_new_tokens else "stop"
            )
            mean_lp, top_lp = _extract_vllm_logprobs(r)
            results.append(GenerationResult(
                text=r.text,
                prompt_tokens=len(output.prompt_token_ids or []),
                completion_tokens=completion_tokens,
                wall_time_s=0.0,  # measured at batch level by the caller
                finish_reason=str(finish) if finish else None,
                mean_logprob=mean_lp,
                answer_top_logprobs=top_lp,
            ))
        return results

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> GenerationResult:
        from vllm import SamplingParams

        top_k_val = int(top_k) if (top_k is not None and top_k > 0) else -1
        merged_stop = self._merge_stop(stop)
        start = time.time()
        sp_kwargs: dict = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": seed,
            "top_p": top_p,
            "top_k": top_k_val,
            "stop": merged_stop,
            "logprobs": logprobs,
        }
        rep_pen = _sampling_repetition_penalty(repetition_penalty)
        if rep_pen is not None:
            sp_kwargs["repetition_penalty"] = rep_pen
        params = SamplingParams(**sp_kwargs)
        outputs = self.llm.generate([prompt], params)
        output = outputs[0]  # RequestOutput — carries prompt_token_ids
        result = output.outputs[0]  # CompletionOutput — carries token_ids / text
        elapsed = time.time() - start
        prompt_tokens = len(output.prompt_token_ids or [])
        completion_tokens = len(result.token_ids or [])
        finish = getattr(result, "finish_reason", None) or (
            "length" if completion_tokens >= max_new_tokens else "stop"
        )
        mean_lp, top_lp = _extract_vllm_logprobs(result)
        return GenerationResult(
            text=result.text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_time_s=elapsed,
            finish_reason=str(finish) if finish else None,
            mean_logprob=mean_lp,
            answer_top_logprobs=top_lp,
        )


def create_backend(model_cfg: dict, dry_run: bool = False) -> BaseBackend:
    """Instantiate a backend and set the active model family.

    When a real model is loaded (vLLM / Transformers), we also register the
    resolved family process-wide so continuation-mode helpers (evaluator,
    adaptive_dialogue) pick up the right think-tags without needing to plumb
    the model name through every call. Baseline overrides from
    `model_family.yaml` (`max_model_len`, `gpu_memory_utilization`) fill in
    when experiment.yaml does not set them explicitly.
    """
    if dry_run:
        return DryRunBackend()

    backend = model_cfg.get("backend", "vllm").lower()
    model_name = model_cfg["name"]

    # Register the active family so downstream helpers (evaluator, dialogue)
    # can resolve tags without explicit plumbing.
    try:
        from model_family import resolve_model_family, set_active_model_family  # noqa: WPS433

        family = set_active_model_family(model_name)
    except Exception:
        family = None  # keep going; ThinkTags defaults still work

    max_model_len = model_cfg.get("max_model_len")
    if max_model_len is None and family is not None:
        max_model_len = family.vllm.max_model_len
    gpu_util = model_cfg.get("gpu_memory_utilization")
    if gpu_util is None and family is not None:
        gpu_util = family.vllm.gpu_memory_utilization
    if gpu_util is None:
        gpu_util = 0.90

    if backend == "transformers":
        return TransformersBackend(model_name, dtype=model_cfg.get("dtype", "auto"))
    if backend == "vllm":
        return VLLMBackend(
            model_name,
            dtype=model_cfg.get("dtype", "auto"),
            max_model_len=max_model_len,
            gpu_memory_utilization=float(gpu_util),
        )
    raise ValueError(f"Unknown backend: {backend}")
