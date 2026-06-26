# Task set curation (experiment_tasks.jsonl)

## Canonical list

All downstream experiments use **`data/experiment_tasks.jsonl`** (100 tasks).  
Do **not** use `data/experiment_tasks.jsonl.pre20swap.bak` — that file is the pre-curation seed-42 intersection snapshot only.

Machine-readable swap details: **`data/meta/task_curation_manifest.json`**.

## Methodology (thesis-ready)

We first formed a candidate set from the GPU task-selection pipeline: tasks that passed verification for the intersection of model sizes, then a **stratified random subsample of 100 tasks** (fixed seed 42) balanced over MATH subjects and difficulty levels. After inspecting verified baselines for the largest model (32B), we **manually curated** this list: **20 tasks were replaced** because their verified chain-of-thought had **exactly three steps**, which is unsuitable for our midpoint-error injection design (no non-degenerate middle step). Replacements were taken from the **remaining 157-task intersection pool**, prioritising the **same MATH level**; where no same-subject candidate existed at that level, we substituted from the pool while keeping level alignment. The resulting set is therefore a **curated intersection subset**, not an i.i.d. random sample from the benchmark.

**Suggested wording (DE):**  
„Die finale Stichprobe von 100 Aufgaben entstammt der schnittmenge-verifizierten MATH-Teilmenge und wurde nach der initialen stratifizierten Zufallsziehung (Seed 42) manuell kuratiert (20 Ersetzungen wegen 3-Schritt-Baselines); sie ist als **kuratierte Schnittmenge** zu berichten, nicht als reine Zufallsstichprobe.“

**Suggested wording (EN):**  
„The final 100-task evaluation set is a **curated subset of the GPU-verified intersection**, obtained after a stratified seed-42 draw and 20 manual replacements to exclude three-step verified baselines; it should not be described as a pure random sample.“
