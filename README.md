# Does Quantization Recovery Generalize to Tool-Calling Reliability?

**Team member:** Hassaan Muzammil  
**Selected track:** Option 1 — Modern Deep Learning Pipeline  

---

## Project Documents

- [Abstract](#abstract)
- [Project Proposal](#project-proposal)
- [Literature Review](#literature-review)
- [AI Audit](#ai-novelty--feasibility-audit)

---

## Abstract

Aggressive post-training quantization makes large language models cheaper to serve but degrades their accuracy. Recover-LoRA addresses this by quantizing a model's MLP gate/up projection layers to 2-bit precision and training small LoRA adapters through knowledge distillation against the full-precision version of the same model, using self-generated synthetic text and no labelled data. It reports recovering 80–95% of the lost accuracy on 9 of 12 general knowledge and reasoning benchmarks.

However, the method has not been evaluated on tool calling: selecting the correct function, populating its arguments, and abstaining when no available function applies. This project tests whether quantization recovery generalizes to that deployment-critical structured capability. We compare five configurations of Qwen3-4B-Instruct-2507: the untouched full-precision model, the damaged quantized model, a reproduction of the published recovery recipe, a variant distilled on tool-calling prompts, and a labelled fine-tuning positive control. We evaluate them on MMLU and HellaSwag, IFEval, and the Berkeley Function Calling Leaderboard (BFCL), while classifying failures by type rather than relying only on aggregate accuracy.

The central claim under test is that a recovery method's headline result is a property of the *(method, evaluation-suite)* pair rather than of the method alone, and that general-purpose benchmarks may overstate what survives quantization for deployment-critical structured capabilities.

--- 

## Project Proposal

### Problem Formulation

### Domain problem

Recover-LoRA reports strong recovery after selective 2-bit quantization, but only on general knowledge and reasoning tasks. A practitioner therefore has no evidence that its reported recovery transfers to tool calling, where the model must make a conditional structured decision:

1. Decide whether a function call is appropriate.
2. Select the correct function from the available schemas.
3. Produce the correct number of arguments with valid names, types, and values.
4. Abstain and answer normally when no function applies.

### Research questions

- **RQ1 — Is recovery capability-dependent?** Does the same general-text recovery adapter restore general benchmarks, instruction following, and tool calling at different rates?
- **RQ2 — Does the distillation prompt domain matter?** With the method, loss, teacher, training budget, rank, and learning rate fixed, does distilling on tool-calling prompts restore tool reliability better than distilling on general text?

### Input and output specification

| Item | Specification |
| --- | --- |
| System input | User query plus available function schemas: name, description, typed parameters, and required fields |
| System output | Structured function call containing a name and JSON arguments, or plain text when no function applies |
| Experimental input | Model checkpoint at a given precision, with or without a LoRA adapter |
| Experimental output | Per-suite scores, chance-corrected recovery, paired statistical results, and failure-channel counts |

### Metrics and success criteria

The primary metric is the **absolute accuracy delta on a chance-corrected scale**:

```text
chance_corrected_score = (raw_score - chance) / (1 - chance)
```

This prevents MMLU's 25% random-guessing floor from artificially inflating recovery relative to BFCL's roughly zero floor.

The secondary metric is the recovery percentage:

```text
recovery = (recovered - quantized) / (full_precision - quantized)
```

Recovery percentage will be reported only when the original damage gap exceeds a pre-registered threshold. It will not be averaged across categories or reported when the denominator is negative or too small.

Every tool-calling failure will be assigned to one of the following channels:

- Hallucinated call
- Abstained when a call was required
- Function name absent from the offered list
- Wrong function selected from the offered list
- Wrong arity
- Argument type error
- Argument value error
- Unparseable output

The project succeeds if RQ1 receives an interpretable answer in either direction. Unequal recovery would show that general benchmarks are insufficient for judging deployment-bound quantized models; equal recovery would extend Recover-LoRA's validated scope to tool calling. If the reproduction does not recover a pre-registered minimum fraction of the general-capability damage gap, the project will invoke a narrower fallback analysis rather than draw conclusions from a failed baseline.

---

### Proposed Technical Approach

#### Experimental configurations

| ID | Configuration | Purpose |
| --- | --- | --- |
| **C1** | Untouched full-precision model | Performance ceiling and recovery denominator |
| **C2** | 2-bit gate/up quantization without recovery | Measures quantization damage |
| **C3** | C2 + LoRA distilled on self-generated general text | Recover-LoRA reproduction; answers RQ1 |
| **C4** | C2 + LoRA distilled on self-generated tool-calling text | Domain-controlled recovery arm; answers RQ2 |
| **C5** | Full-precision model + labelled tool SFT LoRA | Positive control showing that tool capability is movable under the training budget |

C4 remains a distillation experiment, not supervised fine-tuning. The full-precision teacher generates both general-text and tool-calling corpora, so C3 and C4 differ in prompt domain rather than data provenance. Tool samples use hand-authored schemas and are filtered only for structural parseability, not teacher correctness.

#### Model

- **Primary:** Qwen3-4B-Instruct-2507, selected for its 4B scale and native function-calling mode.
- **Optional secondary model:** Granite 4 Micro or Phi-4-mini, subject to compute availability.

All configurations will use the same reasoning mode. The primary experiment is a controlled Qwen case study; the source paper used a different Qwen3-4B variant, so this project will not use its published 80–95% figure as the experimental denominator.

#### Data and evaluation suites

| Suite | Benchmarks | Scoring | Purpose |
| --- | --- | --- | --- |
| **A: General capability** | MMLU, HellaSwag | Log-likelihood ranking | Reproduction check |
| **B: Generative instruction following** | IFEval, prompt-level strict | Free generation | Separates generation/format effects from agentic capability |
| **C: Tool calling** | BFCL v4 live categories, relevance, and irrelevance | Generated calls scored by AST matching | Measures structured tool reliability |

IFEval is essential to the design. Comparing only ranking-scored MMLU/HellaSwag with generative BFCL would confound capability type with evaluation type. Suite A versus B isolates the ranking/generation difference; Suite B versus C isolates non-agentic instruction following from tool calling.

#### Training and analysis

- Selectively quantize the MLP gate/up projections to 2-bit using quantize-dequantize simulation.
- Train LoRA adapters by minimizing KL divergence between full-precision teacher and quantized-student token distributions.
- Hold sample count, step count, loss, teacher, learning rate, and adapter rank fixed between C3 and C4.
- Run a rank sweep at **r = 16** and **r = 64** to test whether low-rank capacity, rather than capability type, explains weak recovery.
- Use two to three seeds per trained configuration to establish a noise floor.
- Use paired item-level **McNemar tests** because every configuration sees the same evaluation items.
- Pre-register equivalence bounds and use TOST when interpreting a null difference between C3 and C4.
- Before training, verify that any top-k logit truncation still includes the behavior-switch tokens that determine whether the model calls a tool or answers in prose.

#### Scope and limitations

- Quantization is simulated rather than kernel-native; therefore, this project makes no throughput or deployment-speed claim.
- The full Recover-LoRA paper and exact recipe must be verified before C3 training begins.
- At 2-bit precision, LoRA capacity may be insufficient to repair the damaged representation; the rank sweep tests this competing explanation.
- Teacher-generated distillation data does not expose the student to its own autoregressive error states.
- The complete five-configuration, three-suite, multi-seed experiment is compute-intensive; the Qwen primary model and load-bearing BFCL categories take priority over the optional second model.

## Literature Review

The survey covers ten recent works from 2024–2026. Recover-LoRA's original paper and extension are treated as one work.

| Work | Year | Main contribution | Relevance to this project |
| --- | --- | --- | --- |
| **Recover-LoRA: Data-Free Accuracy Recovery of Degraded Language Models via Low-Rank Adaptation** (`arXiv:2510.08600`, `arXiv:2606.04238`) | 2025–2026 | Uses synthetic, label-free logit distillation and LoRA to recover selectively quantized LLMs. | Method being reproduced and extended to tool calling. |
| **Flat Score, Amplified Failures: Quantization and Tool-Use Failures in LLM Agents** (`arXiv:2607.27275`) | 2026 | Shows that aggregate task scores can hide large increases in tool-name hallucinations after quantization. | Closest evaluation study, but it uses 4-bit quantization and no weight-level recovery. |
| **Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities** (`arXiv:2505.19433`) | 2025 | Finds that compressed models degrade on agentic tasks and calls for capability-aware distillation. | Establishes the motivation but does not test LoRA-based recovery. |
| **Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery** (`arXiv:2601.20088`) | 2026 | Studies full-parameter recovery and reports that KL loss, the original teacher, and low learning rates work well. | Supplies recovery hyperparameters and a competing claim that narrow-domain data can generalize. |
| **RILQ: Rank-Insensitive LoRA-based Quantization Error Compensation** (`arXiv:2412.01129`) | 2024–2025 | Analyzes why low-rank compensation struggles below 4 bits and proposes a rank-insensitive loss. | Motivates the mandatory rank sweep and an alternative explanation for weak tool recovery. |
| **UPQ: Unifying Block-wise PTQ and Distillation-based QAT for Progressive Quantization toward 2-bit Instruction-Tuned LLMs** (`arXiv:2506.09104`) | 2025 | Finds that general pretraining text may fail to restore instruction following at INT2 and uses distribution matching instead. | Nearest precedent for domain-sensitive recovery of a format-dependent capability. |
| **BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation** (`arXiv:2402.10631`) | 2024 | Applies confidence-aware self-distillation to sub-4-bit models. | Supports analysis of whether a uniform KL loss treats structural and content tokens differently. |
| **When Top-K Misses the Decision: Tool-Call Drift in Multi-Teacher On-Policy Distillation** (`arXiv:2607.07050`) | 2026 | Shows that top-k logits may retain nearly all probability mass while omitting tool-call decision tokens. | Requires a pre-training check that truncation does not remove the behavior under study. |
| **Berkeley Function Calling Leaderboard** | 2025 | Evaluates simple, multiple, parallel, relevance, irrelevance, and multi-turn function calling through AST matching. | Main tool-calling evaluation suite. |
| **Small Reasoning Models are Instruction Followers in Function Calling** (`arXiv:2608.22472`) | 2026 | Reports BFCL results across quantized Qwen3 models and shows that reasoning mode changes quantization sensitivity. | Motivates using live BFCL categories and fixing reasoning mode across configurations. |

### Identified research gap

Prior work establishes that quantization damages agentic behavior and that distillation can recover general capabilities. Other studies disagree on whether recovery-data domain matters, and sub-4-bit work shows that low-rank adapter capacity may be limiting. However, the surveyed work does not apply a quantization-recovery method and directly measure whether it restores tool-calling reliability or whether the same adapter recovers different capability types at different rates.

---

## AI Novelty & Feasibility Audit

### AI Critique Summary

Novelty is real but narrow and incremental rather than foundational. The core method (Recover-LoRA: 2-bit gate/up quantization + KL-distillation LoRA recovery) is not your own — you're reproducing an existing 2025/2026 recipe and extending its *evaluation scope* to tool-calling. The genuine contribution is the RQ1/RQ2 framing: testing whether a recovery method's reported gains transfer across capability types (general knowledge vs. structured tool-calling), and whether recovery-data domain matters when the recipe is otherwise held fixed. This is a legitimate, well-scoped gap — the literature review shows adjacent papers (Flat Score/Amplified Failures, Can Compressed LLMs Truly Act?) establish that quantization hurts agentic/tool behavior, and others (UPQ) show domain-sensitivity of recovery, but none combine LoRA-recovery-method reproduction + tool-calling-specific BFCL evaluation + failure-channel taxonomy in one controlled study. So the "slice" is defensible, but it's an evaluation/ablation contribution on top of someone else's method, not a new algorithm — reviewers at a strong venue could view this as "benchmarking a known technique on a new task" unless the analysis (failure-channel breakdown, chance-corrected scoring, McNemar/TOST statistical rigor) is emphasized as the actual novelty.

### Red ocean / oversaturation risk

Moderate-to-high. Quantization-recovery-via-distillation is an extremely active area (BitDistiller, RILQ, UPQ, QAD-NVFP4, and the base Recover-LoRA paper itself all published within the last \~18 months), so the "recovery method" side of the project sits in a crowded, fast-moving space where your specific angle (tool-calling generalization) could be scooped or already partially covered by a near-future paper — "Flat Score, Amplified Failures" (2607.27275) is uncomfortably close in spirit, even though it doesn't test LoRA recovery. The differentiation needs to be stated explicitly and early in any writeup, or reviewers will read this as "yet another quantization-recovery ablation."

### Feasibility concerns

The five-configuration × three-suite × multi-seed × rank-sweep design (C1–C5, Suites A/B/C, r=16/64, 2–3 seeds) is compute- and engineering-heavy for a single-author, semester-length project — this is the biggest practical risk, not the idea itself. Reproducing Recover-LoRA correctly before touching C3 is itself nontrivial and is explicitly flagged in your own scope section as unverified. The pre-registered success criteria (minimum damage-gap recovery threshold, equivalence bounds for TOST) are good scientific hygiene but raise the bar for what counts as a "successful" result — a failed reproduction triggers a fallback analysis, which is a smart hedge but signals the core pipeline isn't de-risked yet.

### Bottom-line 

Well-posed research question with a clear, citable gap, but the novelty rests on evaluation/analysis design rather than a new method, sits adjacent to a fast-moving red-ocean subfield, and carries meaningful execution risk given the scale of the experimental matrix relative to a single-semester project.

---
