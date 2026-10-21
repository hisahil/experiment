# Methods Notes for the Thesis (documented limitations / design choices)

These notes accompany the pre-full-run fixes in
`.cursor/plans/pre_full_run_fixes_5deca055.plan.md` and the Phase B/C
post-run corrections documented in `.cursor/plans/`. They cover observed
phenomena, documented design divergences, and targeted data fixes that need
to be reflected honestly in the Methods / Limitations section.

## Note 1 — Stop-truncation and reasoning leakage (Punkt 4)

**What.** In the 7B mini run (540 dialogues / 1080 stop+recovery evaluations)
11.3 % of stop rows finished with `finish_reason = length` at the
`stop_max_new_tokens = 500` cap. The corresponding traces show the pattern
Wu et al. (2025, "Interrupted Reasoning") call *reasoning leakage*: after the
force-answer prefill `</think>\n\nTherefore, the final answer is \boxed{`,
the model continues to reason inside the answer segment instead of stopping
after the boxed value.

**Why we do not treat this as a bug.**
- The stop variant is deliberately truncated (a *forced* interrupt), so
  reasoning-leakage-plus-truncation is the *measured* phenomenon, not a
  scoring artefact.
- Our answer extractor takes the **first** `\boxed{…}` in the stop output,
  so an early committed answer is graded even when the model keeps talking.
  This first-boxed rule is applied to **all four** DeepSeek distill sizes
  (1.5B, 7B, 14B, 32B), not to 32B alone. Phase B re-applied it offline via
  `scripts/reextract_stop_answers.py`, followed by selective re-judging and
  re-evaluation wherever stop answers changed. Updated stop rows: 1.5B ~14,
  7B ~14, 14B ~78, 32B ~245 (351 total). Backups are preserved as
  `*.pre_firstboxed` on all four rejudged files. **Rationale:** the stop
  prefill defines the answer commit point; any later `\boxed{}` or continued
  reasoning in the stop segment is non-compliance / recovery-domain behaviour,
  not the graded stop answer.
  Empirically the primary-vs.-as-wrong sensitivity analysis on the mini run
  shows a **0.0 pp** difference on both `susceptibility` and `recovery_gain`
  (`n_truncated=0` in the "flip to wrong" branch): every truncated case is
  already extract-and-wrong, so treating truncated rows as wrong changes
  nothing.
- The force-answer prefill itself is exactly the mitigation Wu et al.
  recommend against reasoning leakage; we already run it (Stop-Prefill v2,
  cf. `scripts/run_perturbed.py` `STOP_PREFILL["math"]`).

**Action.** Report the 11.3 % as an observed model behaviour with a
sensitivity analysis; no cap change. The `stop_max_new_tokens: 500` cap was
raised from 100 for exactly this reason (see `configs/experiment.yaml` inline
comment); lowering it again would remove information without changing the
headline metrics. Where stop-side non-compliance is salient, report
stop-compliance and multi-boxed rates as covariates. The
`final_answer_in_intermediate` marker is unchanged by the first-boxed
re-extract.

**Citation.**

- Wu et al., 2025. *Are Large Reasoning Models Interruptible?* arXiv:2510.11713.

## Note 2 — DeepSeek Usage-Recommendations abgleich (Punkt 5)

Model card: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (also shared with the
14B/32B distill card and the R1 base card). The card's "Usage
Recommendations" list five points; four are followed verbatim, two describe
where and why we deviate.

**Followed.**

- **Temperature 0.5–0.7 (0.6 recommended).** We use `temperature: 0.6`,
  `top_p: 0.95` (`configs/experiment.yaml` model block) — this is exactly
  DeepSeek's benchmark protocol.
- **Multiple samples per query.** Script 1 already draws
  `verification_samples: 5` per task for the majority-vote baseline; the
  main experiment adds 3 seeds per task/condition.
- **Enforce `<think>\n` at the start of every output.** Verified locally
  from the shipped tokenizer template: when `apply_chat_template(...,
  add_generation_prompt=True)` runs, the template appends
  `<｜Assistant｜><think>\n`. The manual fallback prompt (`llm_backend.py`
  `_manual_open_think_prompt`) does the same. In continuation mode the
  reasoning trace is spliced *inside* an already-open `<think>` block, so
  the model cannot bypass the thinking pattern.
- **Model-shipped tokenizer / configs.** `VLLMBackend` and
  `TransformersBackend` both load `AutoTokenizer.from_pretrained(model_name,
  trust_remote_code=True)`.

**Deviation 1 (documented). Recovery-turn system prompt.**

DeepSeek recommends "avoid adding a system prompt; all instructions should
be contained within the user prompt." Our stop and intermediate turns run
without a system prompt (conformant). Only the **recovery** turn attaches a
short formatting instruction (`MATH_RECOVERY_FORMAT_INSTRUCTION` in
`src/llm_backend.py`) so recovery answers reliably contain a `\boxed{…}` for
the deterministic parser — pilot evidence showed markedly higher
ungradable-recovery rates without it.

Because the instruction is applied uniformly to *every* recovery turn, the
between-condition contrasts (adversarial vs. neutral, k-dose, error-type
differences) are untouched by this choice; only the absolute recovery rate
carries a small constant shift versus a strictly conformant configuration.
Changing this before the full run would break comparability with the mini
run without a corresponding scientific gain.

**Deviation 2 (documented). `\boxed{}` directive not in the user prompt.**

DeepSeek recommends the directive "Please reason step by step, and put your
final answer within `\boxed{}`" inside the user prompt. We achieve the same
formatting outcome through three other channels:

- Stop variant: the `\boxed{` prefill *forces* the answer format after the
  interrupt (a stronger version of the empirical suggestion).
- Recovery variant: the same requirement is carried via the recovery system
  prompt described above.
- Baseline (Script 1): the training-time CoT is generated with an explicit
  "Step N:" template so we can segment it at the midpoint for the deletion
  model; `scripts/convert_baselines_to_continuation.py` strips those prefixes
  before the trace is used at inference.

The full-run model input is therefore free CoT with an enforced `\boxed{}`
answer format — semantically equivalent to the recommendation, delivered
through the prefill instead of the user prompt.

**Side note on generation length.** DeepSeek benchmarks use `max_new_tokens
= 32768`. We cap at 12288 (intermediate) / 8192 (recovery) / 500 (stop) to
control cost and force the stop-side interrupt; truncation is monitored
directly (see Note 1).

## Note 3 — 1.5B baseline `<reasoning>` tag contamination (fixed)

**What.** In the 1.5B baseline CoT corpus, 17 of 100 tasks contained
`<reasoning>…</reasoning>` pseudo-tags (not part of the DeepSeek chat
template). These tags propagated into the continuation traces used at
inference, contaminating downstream intermediate and recovery segments.

**Scope.** 918 dialogues (17 tasks × 54 grid cells per task). Of these,
141 intermediate segments were tag-only (no substantive reasoning between
tags), and 265 recovery outputs still carried tag artefacts.

**Root cause.** `scripts/convert_baselines_to_continuation.py` stripped the
Script-1 `Step N:` prefixes but did not remove pseudo-tags from the baseline
CoT before splicing.

**Fix.** Added `TAG_STRIP_RE` to the convert script and ran a targeted GPU
rerun of the 918 affected dialogues only (no full 1.5B re-run). Pre-fix
artefacts are preserved under `data/1.5b/*.pre_tagfix`. The 7B, 14B, and 32B
baseline corpora were checked and are clean for `<reasoning>` pseudo-tags;
no action was taken on those sizes.

**Reporting.** Note the historical contamination and the targeted rerun in
Methods. `final_answer_in_intermediate` is unchanged by this fix.

## Note 4 — 32B Task 1572 stray tag (document only)

**What.** One 32B baseline task —
`math500_test_intermediate_algebra_1572.json` — contains a stray
`</think>` closing tag in the baseline CoT. The 7B and 14B
baselines are otherwise clean for tag issues; among 32B baselines this is the
sole observed contamination (see Note 3 for the separate 1.5B
`<reasoning>` issue).

**Action.** No GPU rerun. The defect affects one contaminated dialogue (54
grid cells) and is documented here as a limitation rather than corrected
post hoc, to avoid a partial re-run for a single task.

## Note 5 — Baseline `correct` flag backfill (acc_clean = 1.0)

**What.** Legacy `baseline_cots.jsonl` exports omitted the top-level `correct`
field even though Script 1 had already verified solvability (≥3/5 correct +
min_steps). Script 4 then re-graded `parsed_answer` vs `gold_answer` (Path B in
`baseline_index.py`), producing format false negatives (e.g. `\frac14` vs
`\frac{1}{4}`) and effective clean accuracy below 1.0 (1.5B: 17/100 tasks;
32B: 10/100).

**Fix.** For all verified rows, set `correct=True` in the four baseline files
(backups: `*.pre_correct_backfill`). This activates Path A: verified baselines
contribute clean accuracy 1.0 by design. `generate_baselines.py` now writes
`correct` on new runs. Re-evaluation only (no GPU, no rejudge).

**Impact.** Primary metrics (Susceptibility, Recovery Gain, H1) unchanged.
`accuracy_drop` now equals `susceptibility` (clean=1.0); RTER and deprecated
recovery-rate metrics shift slightly where they depended on the inflated drop.

## Note 6 — OLMo-3-7B-Think Script-1 baseline repair (post-think fair copy)

**What.** AllenAI Olmo-3-7B-Think emits a hidden `<think>…</think>` draft before
a structured `Step 1:` fair copy. Early Script-1 runs stored the full
`raw_output` (or a malformed `</think#` close) as the baseline CoT, so selected
traces mixed draft prose, close tags, and duplicated step lists. Five MATH
answers also failed deterministic grading on shorthand/currency forms
(`1/4`, `3/4`, `9/19`, `$18.90`, `$36`) despite correct model output.

**Repair (no human rewriting of reasoning).**
1. Offline post-think fair-copy reprocess of all 500 samples via
   `prepare_baseline_raw_text` + `scripts/reprocess_olmo_baselines.py`
   (last `</think>` / `</think#` close; re-segment; recompute degeneracy).
2. Exact DeepSeek longest-correct non-degenerate picker (`majority ≥ 3/5`,
   `min_steps ≥ 3`, all-degenerate fallback unchanged).
3. Thirteen audited **untouched** seed substitutions (stored samples only).
4. Targeted RunPod regeneration of five unresolved tasks with `--base-seed 100`
   and immediate strip-before-grade (`precalculus_1252`, `algebra_2277`,
   `prealgebra_846`, `algebra_661`, `intermediate_algebra_1111`).
5. Independent Claude Sonnet 4.6 re-grade of all samples
   (`scripts/regrade_olmo_baselines.py`); silent `judge_fallback` eliminated.
   Two judge/det disagreements were explicitly adjudicated:
   - `algebra_2735` seed 44: incomplete parsed answer `20` → treat as incorrect
     (deterministic wins).
   - `algebra_2253` seed 45: `\boxed{…}` wrapper vs gold → keep correct
     (judge wins; format-only).
6. Gate 1 structural/provenance audit (`scripts/report_olmo_baseline_gate1.py`)
   before continuation conversion.

**Locked future behaviour.** `baseline_strip_think: true` for OLMo in
`configs/model_families.yaml` while keeping `baseline_use_chat_template: false`
for protocol comparability with the DeepSeek flat-prompt Script-1 runs.
Regression coverage in `tests/test_olmo_baseline_strip.py`.

**Policy.** Never hand-edit model reasoning. Allowed operations are
deterministic parsing, grading normalization, selecting another stored seed,
and new model generation.
