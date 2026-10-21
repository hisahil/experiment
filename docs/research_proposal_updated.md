# Research Proposal

## Working Title

**Robustness and Recovery under Mid-Reasoning Perturbations in Reasoning Language Models**

*Effects of perturbation type, dose, model scale, and model family*

## Abstract

Reasoning language models generate extended textual traces before committing to a final answer. In systems that combine model reasoning with retrieved information, tool outputs, or contributions from other agents, externally produced text may enter this evolving context and influence the eventual decision. Clean benchmark accuracy alone does not show whether a model remains reliable once an otherwise successful reasoning process has been perturbed, nor whether allowing it to continue reasoning repairs or reinforces the disruption.

This thesis investigates these questions through a controlled midpoint-intervention protocol on MATH-500. The study uses a common set of 100 tasks solved by all six evaluated models under a five-sample majority criterion. For each model, an answer-correct baseline trace is truncated at its midpoint and the remaining suffix is deleted. Between one and three pre-generated perturbation steps are then inserted into the open reasoning trace. The perturbations represent three mechanisms: unrelated distraction, premise reframing, and authority-backed advocacy for a wrong decoy answer.

After the final perturbation, two outputs are generated from the same accumulated prefix. The **stop** condition forces an immediate answer under a short token limit, whereas the **recovery** condition permits extended post-perturbation reasoning. This pairing distinguishes immediate susceptibility from the model’s capacity to recover when given additional computation.

The first experiment examines scaling within the DeepSeek-R1-Distill-Qwen family (1.5B, 7B, 14B, and 32B). The second extends the same protocol to GLM-Z1-9B and Olmo-3-7B-Think. Each model contributes 5,400 paired dialogues across perturbation type, perturbation count, perturbation length, and three random seeds. The completed study therefore contains 32,400 dialogues and 64,800 stop/recovery outputs.

The central outcomes are **Susceptibility** (one minus stop accuracy) and **Recovery Gain** (recovery accuracy minus stop accuracy). The implemented analysis also reports clean-normalised token efficiency, recovery persistence, and overcorrection. Task-level inference is used to avoid treating repeated conditions and seeds as independent tasks.

The thesis contributes a reproducible paired stop/recovery protocol, a controlled within-family scaling analysis, and a cross-family extension on matched mathematical tasks. Its central aim is to establish when additional reasoning provides genuine robustness after mid-trace perturbation—and when it merely extends or amplifies an error.

## 1. Motivation and Research Gap

Reasoning models are usually evaluated from a clean prompt to a final answer. This setting omits an important deployment risk: the model may reason within a context that changes while generation is still in progress. A tool can return an incorrect value, retrieval can introduce irrelevant material, or another model can supply a confident but misleading step.

Existing research has examined Chain-of-Thought prompting, misleading hints, self-correction, and the interruptibility of long reasoning traces. However, three distinctions remain insufficiently integrated:

1. **Task competence versus trace robustness:** a model may solve a task cleanly but fail once its successful trace is perturbed.
2. **Immediate commitment versus continued reasoning:** an answer produced at interruption may differ from the answer produced after additional computation on the same context.
3. **Model-specific versus general effects:** a result observed for one model size or family may not generalise to other reasoning models.

This thesis addresses the gap with a matched protocol that begins from answer-correct, model-generated traces; applies typed and dosed midpoint perturbations; pairs stop and recovery outputs on the same prefix; and repeats the measurement across four model sizes and three model families.

The motivating application is interleaved multi-agent and tool-assisted reasoning. The experiment does not reproduce a complete multi-agent system: the inserted text is deliberately unmarked and appears inside an open reasoning trace. The measured construct is therefore robustness to **unattributed mid-reasoning perturbation**, not robustness to an explicitly labelled message from another agent.

## 2. Aim, Research Questions, and Hypotheses

### Aim

To determine how susceptible answer-correct reasoning traces are to controlled midpoint perturbations, how much accuracy is recovered through additional reasoning, and how these effects vary with perturbation mechanism, dose, model scale, and model family.

### Research Questions

- **RQ1 — Susceptibility and recovery:** How often do midpoint perturbations change the immediate answer, and to what extent does continued reasoning restore accuracy?
- **RQ2 — Perturbation characteristics:** How do perturbation type, perturbation count, and perturbation length affect susceptibility and recovery?
- **RQ3 — Model scale:** How do susceptibility and recovery vary across the 1.5B, 7B, 14B, and 32B DeepSeek-R1-Distill-Qwen models?
- **RQ4 — Model family:** Do GLM-Z1-9B and Olmo-3-7B-Think exhibit the same qualitative stop/recovery pattern as the DeepSeek models under the matched protocol?

### Hypotheses

- **H1 (primary):** Recovery accuracy is higher than stop accuracy; equivalently, mean Recovery Gain is positive.
- **H2 (descriptive expectation):** Perturbation mechanism affects the response profile. In particular, an explicit authority-backed decoy is expected to produce high susceptibility and comparatively persistent errors.
- **H3:** Increasing the number of perturbations reduces Recovery Gain by reinforcing the perturbed trajectory.

H1 and the H3 dose trend are the implemented confirmatory tests. H2 is examined through model-specific estimates and diagnostics rather than a direct omnibus test between perturbation types. Length, scale, and family comparisons are secondary or exploratory unless supported by a specified inferential contrast.

## 3. Methodology

### Tasks and Baselines

The confirmatory study is restricted to mathematics and uses the same 100 MATH-500 task identifiers for all six models. For every model-task combination, five baseline outputs are generated. A task is retained when at least three answers are correct and the selected trace contains at least three segmented reasoning steps.

The study therefore estimates robustness conditional on demonstrated task competence rather than unconditional MATH-500 performance. “Answer-correct trace” means that the final answer is verified; it does not imply that every intermediate statement has been human-validated as a correct proof.

### Models

Experiment I evaluates DeepSeek-R1-Distill-Qwen-1.5B, 7B, 14B, and 32B. Experiment II evaluates GLM-Z1-9B-0414 and Olmo-3-7B-Think. All models use the same tasks, but each has its own baseline traces and baseline-conditioned perturbation texts. Cross-model differences are consequently comparative rather than pure causal effects of parameter count or architecture.

### Perturbation Protocol

Each selected baseline is divided into reasoning steps. The first `floor(n/2)` steps are retained and the remaining suffix is deleted. The first perturbation is inserted at this midpoint within the open reasoning trace.

For conditions with multiple perturbations, the model generates one intermediate paragraph before the next perturbation is inserted. The dose manipulation is cumulative: *k* = 1 and *k* = 2 use prefixes of the corresponding three-step perturbation sequence. Thus, *k* represents repeated intervention over an evolving trajectory rather than a single block of repeated text.

After the final perturbation, stop and recovery outputs are generated from the same accumulated prefix:

- **Stop:** a force-answer prefill closes the reasoning block and the model receives a 500-token limit.
- **Recovery:** the model may continue reasoning for up to 8,192 tokens before giving its final answer.

### Perturbation Types

- **Off-topic distraction (`off_topic`):** unrelated first-person reasoning that must not mention the task, its quantities, or a final answer.
- **Premise reframing (`paraphrased_shift`):** a plausible reinterpretation of a value, sign, constraint, or target quantity, without calculation or a final answer.
- **False consensus (`false_consensus`):** an appeal to experts, textbooks, peers, or solution keys in support of an explicit wrong decoy answer, optionally with a task-anchored pseudo-justification.

The perturbations are pre-generated with Claude Sonnet 4.6 and conditioned on each target model’s baseline trace. The construction procedure is common across models, but the realised stimuli are not identical.

### Factorial Design

For each model, the design comprises:

- 100 MATH-500 tasks;
- three perturbation types;
- perturbation counts *k* = 1, 2, and 3;
- short and long perturbations (three versus seven sentences per step);
- three target-model seeds (42, 43, and 44);
- paired stop and recovery outputs.

This produces:

`100 × 3 × 3 × 2 × 3 = 5,400 paired dialogues per model`

and 10,800 stop/recovery records per model. The injection position is fixed at the midpoint. A neutral control and early/late position conditions are outside the executed design.

### Outcomes and Statistical Analysis

The main outcomes are:

- **Susceptibility:** `1 − stop accuracy`;
- **Recovery Gain:** `recovery accuracy − stop accuracy`;
- **RTER:** recovery accuracy per token relative to clean-baseline accuracy per token;
- **Recovery TPR and persistence:** whether initially wrong answers are recovered or remain wrong;
- **Overcorrection:** stop correct but recovery incorrect.

Recovery Gain is the primary inferential outcome. Seeds and conditions are aggregated within tasks, followed by an exact sign/binomial test comparing positive and negative task-level gains. Wilson intervals quantify the task win rate. A task-collapsed linear trend over *k* evaluates H3. Error-type slices test recovery within each type; they do not constitute a direct omnibus H2 comparison.

No Recovery Efficiency Index is proposed. Token use is reported directly and through the implemented descriptive RTER measure rather than presented as a novel scientific index.

### Grading and Quality Assurance

Mathematical answers are first extracted and normalised deterministically. Claude Sonnet 4.6 provides LLM-as-judge grading where required, with disagreements retained for audit. Full-run evaluation requires zero unresolved judge fallbacks.

For stop outputs, the first complete boxed answer after the force-answer prefill is treated as the committed response. A stop output that reaches the token cap without a complete answer is classified as ungradable rather than automatically incorrect; a separate sensitivity analysis evaluates the alternative convention.

## 4. Intended Contributions

1. **Measurement contribution:** a reproducible paired stop/recovery protocol that distinguishes immediate susceptibility from post-perturbation recovery on the same evolving reasoning trace.
2. **Scaling evidence:** a four-size comparison within the DeepSeek-R1-Distill-Qwen family on a common set of solvable mathematical tasks.
3. **Cross-family evidence:** a conceptual replication and extension on GLM-Z1-9B and Olmo-3-7B-Think.
4. **Recovery characterisation:** an analysis of recovery, persistence, and overcorrection as a function of perturbation mechanism, dose, length, and token use.

These are empirical and methodological contributions. The thesis does not claim a new defence, detector, causal theory of model cognition, or universal result across domains.

## 5. Scope and Threats to Validity

- The confirmatory study is restricted to MATH-500.
- Results are conditional on tasks all six models could already solve under the baseline criterion.
- Unmarked insertions may behave differently from explicitly attributed agent messages.
- Without a neutral control, adversarial content cannot be fully separated from the general effect of inserting additional text.
- The fixed midpoint does not establish how early or late interventions behave.
- Model-conditioned traces and stimuli limit causal scale and architecture conclusions.
- Claude Sonnet 4.6 is used for both perturbation generation and judging; deterministic grading and disagreement audits reduce but do not eliminate shared-model bias.
- `Paraphrased_shift` can occasionally be constructive, whereas `false_consensus` explicitly reveals a decoy. The types must therefore be interpreted as distinct operational perturbations rather than levels of one severity scale.
- Baseline cleaning and targeted reruns are disclosed as protocol refinements.

## 6. Impact Statement

The study addresses reliability risks in systems that combine long-form model reasoning with external text. Its findings can inform interrupt-and-answer interfaces, multi-agent workflows, and tool-use pipelines by showing that clean task competence does not guarantee robustness to mid-trace perturbation and that additional reasoning can help without being uniformly safe.

The work uses no personal or sensitive data. Its main risks are overgeneralisation and anthropomorphic interpretation of generated reasoning traces. These are mitigated by restricting claims to the tested protocol, describing the outputs as textual traces rather than private cognition, and reporting exclusions and preprocessing decisions transparently.

## 7. Proposed Thesis Structure

### Chapter 1: Introduction

This chapter introduces the reliability problem posed by mid-reasoning perturbations. It motivates the study through interleaved agent and tool contexts, defines the research gap, presents the research questions and hypotheses, and states the scope and contributions of the thesis.

1.1 Research Context and Motivation  
1.2 Problem Statement and Research Gap  
1.3 Aim, Research Questions, and Hypotheses  
1.4 Scope and Contributions  
1.5 Thesis Structure

### Chapter 2: Background and Related Work

This chapter establishes only the theoretical and empirical foundations needed for the study. It first explains generated reasoning traces, then reviews robustness to interventions within those traces. Interruptibility and self-correction are treated together because they directly motivate the paired stop/recovery design. Multi-agent and tool-assisted systems are discussed as an application context rather than as a separate architecture survey. The chapter closes by synthesising the literature into the final research gap.

2.1 Reasoning LLMs and Generated Reasoning Traces  
2.1.1 Chain-of-Thought Prompting and Extended Reasoning  
2.1.2 Reasoning Traces, Answer Commitment, and Interpretive Limits  
2.2 Robustness to Mid-Reasoning Perturbations  
2.2.1 Misleading Hints, Decoys, and Intermediate Interventions  
2.2.2 Error Propagation in Interleaved Agent and Tool Contexts  
2.3 Interruptibility, Self-Correction, and Recovery  
2.3.1 Forced-Answer Behaviour and Reasoning Leakage  
2.3.2 Recovery, Overthinking, and Overcorrection  
2.4 Evaluating Mathematical Reasoning and LLM-as-Judge Reliability  
2.5 Synthesis and Research Gap

### Chapter 3: Methodology

This chapter describes the executed experimental design. It explains task selection, model-specific baseline construction, perturbation generation, the midpoint continuation protocol, the factorial design, and the stop/recovery conditions. It concludes with the outcome measures, statistical analysis, grading procedures, and quality controls.

3.1 Study Overview  
3.2 Dataset and Common MATH-500 Task Set  
3.3 Model Selection and Inference Configuration  
3.3.1 DeepSeek Scale Series  
3.3.2 GLM and Olmo Cross-Family Extension  
3.4 Baseline Generation and Verification  
3.5 Perturbation Framework  
3.5.1 Perturbation Taxonomy  
3.5.2 Perturbation Count, Length, and Model Conditioning  
3.6 Midpoint Deletion and Continuation Protocol  
3.6.1 Intermediate Model Generations  
3.6.2 Stop and Recovery Conditions  
3.7 Factorial Design and Sample Size  
3.8 Outcome Measures and Statistical Analysis  
3.8.1 Susceptibility and Recovery Gain  
3.8.2 RTER, Persistence, and Overcorrection  
3.8.3 Confirmatory and Descriptive Analyses  
3.9 Grading and Quality Assurance  
3.10 Pilot, Protocol Refinements, and Reproducibility

### Chapter 4: Results

This chapter presents the empirical results in the order of the research questions. It first establishes data completeness and grading quality, then reports overall susceptibility and recovery. Subsequent sections examine perturbation characteristics, DeepSeek scaling, and the GLM/Olmo extension before characterising recovery outcomes and robustness checks.

4.1 Data Completeness and Quality Gates  
4.2 Overall Susceptibility and Recovery (RQ1 / H1)  
4.3 Effects of Perturbation Characteristics (RQ2)  
4.3.1 Perturbation Type  
4.3.2 Perturbation Count and Dose Trend (H3)  
4.3.3 Perturbation Length  
4.4 DeepSeek Scaling Analysis (RQ3)  
4.5 GLM and Olmo Cross-Family Extension (RQ4)  
4.6 Recovery Pattern Characterisation  
4.6.1 Successful Recovery, Persistence, and Overcorrection  
4.6.2 Decoy Anchoring  
4.6.3 Token Use and RTER  
4.7 Robustness and Sensitivity Analyses

### Chapter 5: Discussion

This chapter interprets the findings in relation to the research questions and prior literature. It explains the distinction between immediate susceptibility and recovery, examines the roles of perturbation mechanism and dose, and discusses scale and family differences. It then derives practical implications and evaluates the study’s limitations.

5.1 Principal Findings and Stop/Recovery Interpretation  
5.2 Perturbation Mechanism, Dose, and Length  
5.3 Model Scale and Cross-Family Generalisation  
5.4 Practical Implications for Agent and Tool Pipelines  
5.5 Limitations and Threats to Validity  
5.6 Future Work

### Chapter 6: Conclusion

This chapter provides a concise synthesis of the thesis. It answers each research question directly, summarises the methodological and empirical contributions, and closes with the broader implications for evaluating reasoning-model reliability.

6.1 Answers to the Research Questions  
6.2 Summary of Contributions  
6.3 Closing Remarks

### Bibliography

The bibliography will include work on Chain-of-Thought, reasoning-model robustness, interruptibility, self-correction, MATH-500, LLM-as-judge reliability, and the reports and model cards for DeepSeek-R1, GLM-Z1, and Olmo-3.

### Appendices

Appendix A: Prompts, Perturbation Specifications, and Model Configurations  
Appendix B: Full Results, Grading, and Sensitivity Analyses  
Appendix C: Representative Reasoning Traces and Documented Repairs  
Appendix D: Reproducibility Information

## 8. Current Project Status

Experimental execution is complete for all six included models. Each model has 5,400 paired dialogues and 10,800 output records with zero unresolved judge fallbacks.

Current summaries show that Susceptibility ranges from approximately 0.50 to 0.74 and Recovery Gain from approximately 0.23 to 0.38. Recovery Gain is positive for every model (`p < 1e-4` in the overall task-level tests). Within the DeepSeek family, Susceptibility decreases monotonically from 1.5B to 32B. Recovery Gain decreases with increasing *k* in all four DeepSeek models, while the corresponding trend is weaker and not statistically clear for GLM and Olmo.

## Core References to Develop

- von Recum et al., *Are Reasoning LLMs Robust?*
- Wu et al., *Are Large Reasoning Models Interruptible?*
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
- MATH / MATH-500 documentation
- DeepSeek-R1, GLM-Z1, and Olmo-3 model reports and model cards
- Literature on intrinsic self-correction and LLM-as-judge reliability
