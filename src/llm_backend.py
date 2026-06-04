"""LLM backends for local dry-run and GPU inference."""

from __future__ import annotations

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


Message = dict[str, str]

COLLABORATING_AGENT_TAG = "[Note from collaborating agent]:"


def format_collaborating_note(injection_steps: list[str]) -> str:
    body = "\n".join(injection_steps)
    return f"{COLLABORATING_AGENT_TAG}\n{body}"


def build_perturbed_messages(
    problem: str,
    baseline_prefix: str,
    injection_steps: list[str],
    *,
    reasoning_model: bool = False,
    trace_in_user_turn: bool = False,
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
    note = format_collaborating_note(injection_steps)

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


class BaseBackend:
    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> GenerationResult:
        raise NotImplementedError

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GenerationResult]:
        """Submit multiple prompts; base implementation is sequential."""
        return [self.generate(p, max_new_tokens, temperature, top_p) for p in prompts]

    def generate_chat(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        prefill: str | None = None,
    ) -> GenerationResult:
        prompt = self.format_messages(messages, prefill=prefill)
        return self.generate(prompt, max_new_tokens, temperature, top_p)

    def generate_chat_batch(
        self,
        items: list[tuple[list[Message], str | None]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GenerationResult]:
        """
        Batch variant of generate_chat.

        items: list of (messages, prefill) pairs.
        Returns one GenerationResult per item, in the same order.
        Prefill text is prepended to results when supplied.
        """
        prompts = [self.format_messages(msgs, prefill=pre) for msgs, pre in items]
        raw_results = self.generate_batch(prompts, max_new_tokens, temperature, top_p)
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

    def format_messages(self, messages: list[Message], *, prefill: str | None = None) -> str:
        return messages_to_prompt(None, messages, prefill=prefill)

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GenerationResult]:
        """Dry-run batch: call generate() for each prompt (no GPU needed)."""
        return [self.generate(p, max_new_tokens, temperature, top_p) for p in prompts]

    def generate_chat(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        prefill: str | None = None,
    ) -> GenerationResult:
        prompt = self.format_messages(messages, prefill=prefill)
        result = self.generate(prompt, max_new_tokens, temperature, top_p)
        if prefill:
            result = GenerationResult(
                text=prefill + result.text,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                wall_time_s=result.wall_time_s,
                finish_reason=result.finish_reason,
            )
        return result

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> GenerationResult:
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

    def format_messages(self, messages: list[Message], *, prefill: str | None = None) -> str:
        return messages_to_prompt(self.tokenizer, messages, prefill=prefill)

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GenerationResult]:
        """Sequential fallback — TransformersBackend does not support true batching."""
        import warnings
        warnings.warn(
            "TransformersBackend.generate_batch() runs prompts sequentially. "
            "Use VLLMBackend for true GPU batching.",
            stacklevel=2,
        )
        return [self.generate(p, max_new_tokens, temperature, top_p) for p in prompts]

    def generate_chat(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        prefill: str | None = None,
    ) -> GenerationResult:
        prompt = self.format_messages(messages, prefill=prefill)
        result = self.generate(prompt, max_new_tokens, temperature, top_p)
        if prefill:
            result = GenerationResult(
                text=prefill + result.text,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                wall_time_s=result.wall_time_s,
                finish_reason=result.finish_reason,
            )
        return result

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> GenerationResult:
        import torch

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
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        elapsed = time.time() - start
        completion_tokens = int(generated.shape[-1])
        finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_time_s=elapsed,
            finish_reason=finish_reason,
        )


class VLLMBackend(BaseBackend):
    def __init__(self, model_name: str, dtype: str = "auto"):
        from vllm import LLM

        self.llm = LLM(model=model_name, trust_remote_code=True, dtype=dtype)
        self.model_name = model_name
        self.tokenizer = None
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            pass

    def format_messages(self, messages: list[Message], *, prefill: str | None = None) -> str:
        return messages_to_prompt(self.tokenizer, messages, prefill=prefill)

    def generate_chat(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        prefill: str | None = None,
    ) -> GenerationResult:
        prompt = self.format_messages(messages, prefill=prefill)
        result = self.generate(prompt, max_new_tokens, temperature, top_p)
        if prefill:
            result = GenerationResult(
                text=prefill + result.text,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                wall_time_s=result.wall_time_s,
                finish_reason=result.finish_reason,
            )
        return result

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[GenerationResult]:
        """Submit all prompts in one vLLM call for maximum GPU utilisation."""
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        outputs = self.llm.generate(prompts, params)
        results: list[GenerationResult] = []
        for output in outputs:
            r = output.outputs[0]
            completion_tokens = len(r.token_ids or [])
            finish = getattr(r, "finish_reason", None) or (
                "length" if completion_tokens >= max_new_tokens else "stop"
            )
            results.append(GenerationResult(
                text=r.text,
                prompt_tokens=len(output.prompt_token_ids or []),
                completion_tokens=completion_tokens,
                wall_time_s=0.0,  # measured at batch level by the caller
                finish_reason=str(finish) if finish else None,
            ))
        return results

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> GenerationResult:
        from vllm import SamplingParams

        start = time.time()
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        outputs = self.llm.generate([prompt], params)
        output = outputs[0]  # RequestOutput — carries prompt_token_ids
        result = output.outputs[0]  # CompletionOutput — carries token_ids / text
        elapsed = time.time() - start
        prompt_tokens = len(output.prompt_token_ids or [])
        completion_tokens = len(result.token_ids or [])
        finish = getattr(result, "finish_reason", None) or (
            "length" if completion_tokens >= max_new_tokens else "stop"
        )
        return GenerationResult(
            text=result.text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_time_s=elapsed,
            finish_reason=str(finish) if finish else None,
        )


def create_backend(model_cfg: dict, dry_run: bool = False) -> BaseBackend:
    if dry_run:
        return DryRunBackend()
    backend = model_cfg.get("backend", "vllm").lower()
    if backend == "transformers":
        return TransformersBackend(model_cfg["name"], dtype=model_cfg.get("dtype", "auto"))
    if backend == "vllm":
        return VLLMBackend(model_cfg["name"], dtype=model_cfg.get("dtype", "auto"))
    raise ValueError(f"Unknown backend: {backend}")
