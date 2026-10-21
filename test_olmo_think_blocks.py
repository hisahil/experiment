"""
Live generation test: OLMo 3 7B Think — chat template with think blocks.
"""

import sys
import time
from datetime import datetime, timezone

MODEL_NAME = "allenai/OLMo-3-7B-Think"

TASKS = [
    {
        "id": "easy_linear",
        "question": "What is the value of x if 3x + 7 = 22?",
    },
    {
        "id": "quadratic",
        "question": "Find all values of x that satisfy x^2 - 5x + 6 = 0.",
    },
]

def apply_vllm_patches(log):
    """Apply all vLLM 0.6.6 compatibility patches for OLMo 3."""
    
    # ── Patch 1: Register Olmo3ForCausalLM via ModelRegistry API ────────
    log("  [patch 1] Registering Olmo3ForCausalLM via ModelRegistry...")
    from vllm.model_executor.models.registry import ModelRegistry
    if "Olmo3ForCausalLM" not in ModelRegistry.models:
        ModelRegistry.register_model(
            "Olmo3ForCausalLM",
            "vllm.model_executor.models.olmo2:Olmo2ForCausalLM",
        )
        log("    Registered Olmo3ForCausalLM -> olmo2:Olmo2ForCausalLM")
    else:
        log("    Already registered")

    # ── Patch 2: rope_scaling validation ────────────────────────────────
    log("  [patch 2] Patching rope_scaling validation...")
    import vllm.transformers_utils.config as vllm_config_mod
    _original_patch = vllm_config_mod.patch_rope_scaling
    def _patched_rope_scaling(config):
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling and isinstance(rope_scaling, dict):
            if "sliding_attention" in rope_scaling or "full_attention" in rope_scaling:
                return
        return _original_patch(config)
    vllm_config_mod.patch_rope_scaling = _patched_rope_scaling

    # ── Patch 3: _get_and_verify_max_len rope_scaling access ────────────
    log("  [patch 3] Patching _get_and_verify_max_len...")
    import vllm.config as vllm_main_config
    _original_get_max_len = vllm_main_config._get_and_verify_max_len
    def _patched_get_max_len(hf_config, max_model_len, *args, **kwargs):
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling and isinstance(rope_scaling, dict):
            if "sliding_attention" in rope_scaling or "full_attention" in rope_scaling:
                full = rope_scaling.get("full_attention", {})
                flat = dict(full)
                flat.setdefault("rope_type", "yarn")
                flat.setdefault("factor", 8.0)
                flat.setdefault("original_max_position_embeddings", 8192)
                orig = hf_config.rope_scaling
                hf_config.rope_scaling = flat
                try:
                    result = _original_get_max_len(hf_config, max_model_len, *args, **kwargs)
                finally:
                    hf_config.rope_scaling = orig
                return result
        return _original_get_max_len(hf_config, max_model_len, *args, **kwargs)
    vllm_main_config._get_and_verify_max_len = _patched_get_max_len
    
    log("  All patches applied.")


def main():
    log_path = "/root/msc_thesis_experiment/logs/olmo_think_block_test.txt"
    out_lines = []

    def log(msg=""):
        print(msg, flush=True)
        out_lines.append(msg)

    log(f"{'='*80}")
    log(f"OLMo 3 7B Think — Live Think-Block Generation Test")
    log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log(f"Model: {MODEL_NAME}")
    log(f"{'='*80}")

    # ── Load tokenizer ──────────────────────────────────────────────────
    log("\n[1/4] Loading tokenizer...")
    t0 = time.time()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    log(f"  Tokenizer loaded in {time.time()-t0:.1f}s")
    log(f"  Vocab size: {tokenizer.vocab_size}")
    log(f"  Chat template present: {tokenizer.chat_template is not None}")

    special_tokens = [t for t in tokenizer.all_special_tokens if 'think' in t.lower()]
    log(f"  Think-related special tokens: {special_tokens or 'none found'}")

    log(f"\n  >>> CHAT TEMPLATE <<<")
    log(f"  ```")
    log(f"  {tokenizer.chat_template}")
    log(f"  ```")

    # ── Apply vLLM patches ──────────────────────────────────────────────
    log("\n[2/4] Applying vLLM 0.6.6 compatibility patches...")
    apply_vllm_patches(log)

    # ── Load vLLM engine ────────────────────────────────────────────────
    log("\n[3/4] Loading vLLM engine...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        hf_overrides={"rope_theta": 500000},
        disable_log_stats=True,
    )
    log(f"  vLLM engine loaded in {time.time()-t0:.1f}s")

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
    )

    # ── Generate for each task ──────────────────────────────────────────
    log(f"\n[4/4] Generating responses for {len(TASKS)} tasks...")
    log(f"  Sampling: temperature=0.6, top_p=0.95, max_tokens=2048")

    for i, task in enumerate(TASKS):
        log(f"\n{'─'*80}")
        log(f"TASK {i+1}/{len(TASKS)}: {task['id']}")
        log(f"{'─'*80}")
        log(f"Question: {task['question']}")

        messages = [
            {"role": "user", "content": task["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        log(f"\n>>> RENDERED PROMPT (apply_chat_template) <<<")
        log(f"```")
        log(prompt)
        log(f"```")
        log(f"  Prompt length (chars): {len(prompt)}")
        prompt_tokens = tokenizer.encode(prompt)
        log(f"  Prompt length (tokens): {len(prompt_tokens)}")

        t0 = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.time() - t0
        result = outputs[0]
        generated_text = result.outputs[0].text
        gen_tokens = len(result.outputs[0].token_ids)
        finish_reason = result.outputs[0].finish_reason

        log(f"\n>>> RAW MODEL OUTPUT (complete, {gen_tokens} tokens, {elapsed:.1f}s) <<<")
        log(f"  Finish reason: {finish_reason}")
        log(f"  Tokens/second: {gen_tokens/elapsed:.1f}")
        log(f"```")
        log(generated_text)
        log(f"```")

        # ── Analyze structure ───────────────────────────────────────────
        log(f"\n>>> STRUCTURAL ANALYSIS <<<")

        has_think_open = "<think>" in generated_text
        has_think_close = "</think>" in generated_text
        log(f"  Contains <think>:  {has_think_open}")
        log(f"  Contains </think>: {has_think_close}")

        if not has_think_open and has_think_close:
            log(f"  NOTE: <think> was injected by chat template. Model continues from inside think block.")
            think_end = generated_text.index("</think>")
            think_content = generated_text[:think_end]
            post_think = generated_text[think_end + len("</think>"):].strip()

            log(f"  Think content length: {len(think_content)} chars")
            log(f"  Post-think content length: {len(post_think)} chars")

            log(f"\n  >>> THINK BLOCK CONTENT (inside <think>...</think>) <<<")
            log(f"  ```")
            for line in think_content.split('\n'):
                log(f"  {line}")
            log(f"  ```")

            log(f"\n  >>> POST-THINK CONTENT (after </think>) <<<")
            log(f"  ```")
            for line in post_think.split('\n'):
                log(f"  {line}")
            log(f"  ```")

            has_step_n = any(f"Step {n}" in think_content or f"step {n}" in think_content for n in range(1, 10))
            has_boxed = "\\boxed{" in post_think or "\\boxed{" in generated_text
            has_final_answer = "final answer" in post_think.lower()
            has_therefore = "therefore" in post_think.lower()

            log(f"\n  Pattern detection:")
            log(f"    'Step N:' in think block: {has_step_n}")
            log(f"    '\\\\boxed{{' anywhere: {has_boxed}")
            log(f"    'final answer' in post-think: {has_final_answer}")
            log(f"    'therefore' in post-think: {has_therefore}")

        elif has_think_open and has_think_close:
            think_start = generated_text.index("<think>")
            think_end = generated_text.index("</think>")
            think_content = generated_text[think_start + len("<think>"):think_end]
            post_think = generated_text[think_end + len("</think>"):].strip()

            log(f"  Think block position: chars {think_start}-{think_end}")
            log(f"  Think content length: {len(think_content)} chars")
            log(f"  Post-think content length: {len(post_think)} chars")

            log(f"\n  >>> THINK BLOCK CONTENT <<<")
            log(f"  ```")
            for line in think_content.split('\n'):
                log(f"  {line}")
            log(f"  ```")

            log(f"\n  >>> POST-THINK CONTENT <<<")
            log(f"  ```")
            for line in post_think.split('\n'):
                log(f"  {line}")
            log(f"  ```")

            has_step_n = any(f"Step {n}" in think_content or f"step {n}" in think_content for n in range(1, 10))
            has_boxed = "\\boxed{" in post_think or "\\boxed{" in generated_text
            has_final_answer = "final answer" in post_think.lower()
            has_therefore = "therefore" in post_think.lower()

            log(f"\n  Pattern detection:")
            log(f"    'Step N:' in think block: {has_step_n}")
            log(f"    '\\\\boxed{{' anywhere: {has_boxed}")
            log(f"    'final answer' in post-think: {has_final_answer}")
            log(f"    'therefore' in post-think: {has_therefore}")

        elif has_think_open and not has_think_close:
            log(f"  WARNING: Think block opened but never closed (hit max_tokens).")
            think_start = generated_text.index("<think>")
            think_content = generated_text[think_start + len("<think>"):]
            log(f"  Think content (truncated): {len(think_content)} chars")
            log(f"  ```")
            for line in think_content.split('\n'):
                log(f"  {line}")
            log(f"  ```")
        else:
            log(f"  No think blocks found. Model produced flat output.")
            log(f"  (Note: <think> was injected by chat template at end of prompt)")
            log(f"  Full output:")
            log(f"  ```")
            for line in generated_text.split('\n'):
                log(f"  {line}")
            log(f"  ```")

    # ── Summary ─────────────────────────────────────────────────────────
    log(f"\n{'='*80}")
    log(f"COMPATIBILITY ASSESSMENT")
    log(f"{'='*80}")
    log(f"Key questions for continuation pipeline compatibility:")
    log(f"  1. Does model produce <think>...</think> blocks?")
    log(f"  2. Is reasoning inside think block free-form or structured?")
    log(f"  3. What format is the post-think answer?")
    log(f"  4. Can we inject corrupted CoT into an open think block?")
    log(f"{'='*80}")

    with open(log_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    log(f"\nLog saved to: {log_path}")

if __name__ == "__main__":
    main()
