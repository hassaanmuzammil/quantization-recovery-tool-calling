# Literature & SOTA Survey

Ten works published 2024–2026, grouped by their relationship to this project.

---

## 1\. The method being extended

### Recover-LoRA — Data-Free Accuracy Recovery of Degraded Language Models via Low-Rank Adaptation

[arXiv 2606.04238](https://arxiv.org/abs/2606.04238) (extension) and [arXiv 2510.08600](https://arxiv.org/abs/2510.08600) (original, EMNLP 2025 Industry Track) **Verification: search-verified — full read pending before implementation**

Quantizes MLP gate/up projections to 2-bit while leaving attention and embeddings at higher precision, then trains LoRA adapters on the quantized layers via logit distillation (KL divergence) against the full-precision teacher. Training data is synthetic text the model generates itself, so the method requires no external dataset and no labels. Reports 80–95% recovery on 9 of 12 general benchmarks for Qwen3-4B using approximately 10k samples.

**Details still to confirm.** The following are reported second-hand and have not been verified against the papers themselves: that the extension differs from the original in adapter placement (gate/up rather than attention K/V), initialization (standard LoRA rather than OLoRA), and sample efficiency (10k vs 90–120k); and the EMNLP 2025 Industry Track venue for the original. Confirming these, along with the exact synthetic-data procedure and which twelve benchmarks are used, is a prerequisite for implementation — C3 is defined as a faithful reproduction of this recipe.

**Relevance:** this is the method under test. Its explicit scoping to general capability, and its listing of broader capability validation as future work, is the gap this project addresses. The reported figure of 80–95% recovery on *9 of 12* benchmarks implies three under-recovered; identifying which three, and whether they share a property, is an open item that may bear directly on RQ1.

---

## 2\. Quantization and agentic capability — the closest prior work

### Flat Score, Amplified Failures: Quantization and Tool-Use Failures in LLM Agents

[arXiv 2607.27275](https://arxiv.org/abs/2607.27275) **Verification: read in full**

Evaluates Gemma-4 and Qwen-3.6 families on τ²-bench at BF16/FP8/INT4 across ten model-domain cells. Finds no significant change in final task score from 16-bit to 4-bit — seven cells are statistically equivalent to zero by TOST at ±7.5 points — while the underlying process degrades severely: tool-name hallucination in the worst cell rises from 19.5% to 38.3% of calls, a 2.5× increase in event volume.

Two mechanisms explain the gap. First, the failure *set* is precision-invariant: the distribution over which tool names get invented correlates 0.97 between BF16 and INT4, and only 0.18% of INT4 hallucinations name a tool the BF16 model never invented, so quantization amplifies pre-existing failures rather than creating new ones. Second, the benchmark's ten-error budget absorbs the extra failures; tightening it to two re-exposes a 16.7-point gap. Susceptibility tracks full-precision failure propensity rather than model size or architecture, explained via a logit-margin account in which quantization noise flips only thin-margin decisions.

**Relevance and differentiation:** the closest competitor. This project differs in bit depth (2-bit selective vs 4-bit AWQ), model scale (4B vs 26–35B), benchmark (BFCL vs τ²-bench), and most importantly in applying weight-level recovery — their mitigations are prompt-level only. Their closing argument that the amplification mechanism should generalize to any weight perturbation, including low-rank compression and distillation drift, is precisely the opening this project occupies.

### Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities

[arXiv 2505.19433](https://arxiv.org/abs/2505.19433) (Dong et al., ICML 2025\) **Verification: search-verified**

Establishes empirically that compressed models degrade on agentic tasks, and calls for distillation approaches that balance abstract reasoning against practical agentic capability. Uses aggregate metrics and applies no LoRA-based recovery.

**Relevance:** states the gap this project fills. Cited as motivation rather than as a competing method.

---

## 3\. Distillation-based recovery — method and hyperparameters

### Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery (NVIDIA)

[arXiv 2601.20088](https://arxiv.org/abs/2601.20088) **Verification: read in full**

Full-parameter distillation recovery for NVFP4 quantization. Establishes that KL divergence outperforms MSE on logits, that the original model makes a better teacher than a larger sibling, that temperature T=1 for both teacher and student is correct, and that learning rates should sit between 1e-6 and 1e-5 for SFT-converged models.

Its Table 4 is the specific prior claim this project's RQ2 tests: QAD using only code data recovered strong math performance (AIME24 71.0 for code-only vs 71.7 for math+code), attributed to the teacher's output distributions encoding knowledge about all domains even when input data is narrow. Table 5 qualifies this — a monotone gradient runs across data sources (random tokens 60.0, BOS-generated 60.9, correct-only 61.6, SFT data 62.0 on AIME25, against a PTQ baseline of 58.7) — so data quality does matter, modestly, at their damage level.

**Relevance and differentiation:** the damage gap they recover from is small (PTQ 69.4 vs BF16 73.0), training is full-parameter rather than rank-limited, and both domains are reasoning capabilities within one model's specialization. This project tests whether the property survives at 2-bit, through a low-rank adapter, for a structured capability. Technical report, not peer-reviewed.

### RILQ: Rank-Insensitive LoRA-based Quantization Error Compensation

[arXiv 2412.01129](https://arxiv.org/abs/2412.01129) (AAAI 2025\) **Verification: search-verified, quoted claims confirmed**

Names this project's setting: *LoRA-based Quantization Error Compensation* (LQEC). Reports that LQEC has underperformed in sub-4-bit scenarios with no prior investigation into understanding the limitation, diagnoses it through rank analysis, and proposes a model-wise activation discrepancy loss that is rank-insensitive. Their Table 4 shows SVD-based compensation improving only marginally from rank 16 to rank 256\.

**Relevance:** the most serious alternative explanation for this project's RQ1. If recovery is worse for tool calling than for general capability, "low-rank adapters have limited capacity at 2-bit and the harder task exhausts it first" competes directly with "recovery is capability-dependent." This is why a rank sweep is a required component rather than optional depth.

### UPQ: Unifying Block-wise PTQ and Distillation-based QAT for Progressive Quantization toward 2-bit Instruction-Tuned LLMs

[arXiv 2506.09104](https://arxiv.org/abs/2506.09104) **Verification: search-verified, quoted claims confirmed**

Argues that minimizing next-token-prediction loss on a pre-training corpus during INT2 QAT of instruction-tuned models often fails to recover instruction-following capability, because pre-training corpora consist of general text rather than instruction-response pairs. Introduces INT2 Distill-QAT to mimic the FP16 model's token-level distribution instead, and evaluates on MMLU alongside IFEval.

**Relevance:** the nearest neighbour to RQ2 — same bit width, and a format-sensitive capability one step removed from tool calling. Differentiation: they change the *loss* (general-text NTP → distribution matching) while this project holds the loss fixed and varies only the prompt domain. Also the direct precedent for pairing MMLU with IFEval.

### BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation

[arXiv 2402.10631](https://arxiv.org/abs/2402.10631) **Verification: search-verified**

Sub-4-bit quantization with self-distillation and a confidence-aware KL objective, including analysis of per-token teacher confidence across generation and reasoning tasks.

**Relevance:** the closest published precedent for self-distillation as a recovery mechanism at this project's bit depth, and the proper source for a claim an early AI critique attributed to a non-existent paper — that teacher confidence varies sharply by token type, and that a uniform KL objective may therefore distribute gradient unevenly across structural and content positions. Relevant to whether distillation reaches the tokens that determine tool-call validity.

---

## 4\. The distillation loss and tool-call decisions

### When Top-K Misses the Decision: Tool-Call Drift in Multi-Teacher On-Policy Distillation

[arXiv 2607.07050](https://arxiv.org/abs/2607.07050) **Verification: search-verified, quoted claims confirmed**

Finds that a response teacher's top-32 logits retain 99.99% of probability mass yet contain the tool-call behaviour-switch token on only 0.4% of 1,500 audited prompts; even top-256 covers only 52.2%. Because omitted logits receive zero direct gradient under a truncated objective, the decision to emit a tool call versus answer in prose can go entirely unsupervised. Also reports schema drift — checkpoints producing invalid tool-call strings despite a correct target format.

**Relevance:** a methodological landmine. If memory pressure forces top-K truncation of the distillation loss, the exact decision boundary this project studies may receive no gradient, making any C3-vs-C4 result an artifact of the loss implementation rather than the prompt domain. Must be checked empirically before training.

---

## 5\. Evaluation

### Berkeley Function Calling Leaderboard

[Patil et al., ICML 2025](https://proceedings.mlr.press/v267/patil25a.html) **Verification: search-verified; harness run directly and results validated**

The evaluation benchmark. Categories span simple, multiple, parallel and parallel-multiple function calls, relevance and irrelevance detection, and multi-turn interaction, across both curated (non-live) and user-contributed (live) function schemas. Scored by abstract syntax tree matching against ground-truth calls.

### Small Reasoning Models are Instruction Followers in Function Calling

[arXiv 2608.22472](https://arxiv.org/abs/2608.22472) **Verification: read in full**

Measures Q4KM quantization impact on BFCL v3 across the Qwen3 family. Two findings directly shaped this project's design. First, non-live categories sit at 90–96% for Qwen3-4B, which is too close to ceiling for damage or recovery to be measurable — motivating the choice of live categories here. Second, reasoning mode dramatically changes quantization sensitivity: Qwen3-0.6B in no-think mode falls 62.3 → 16.8 under quantization while think mode falls only 76.4 → 69.2, attributed to the reasoning trace acting as error correction for weight precision loss. This makes thinking mode a confound that must be fixed identically across all configurations.

Caveat: a v1 preprint with visible quality issues (several table cells missing, some internally inconsistent numbers). Cited as a design-informing data point, not as a load-bearing baseline.

---

## Summary of the gap

Quantization damages agentic capability (Dong et al.; Flat Score). Recovery methods restore general capability (Recover-LoRA; QAD). Recovery-data domain matters for format-sensitive capabilities at 2-bit (UPQ), or does not matter much at milder damage levels (QAD). Low-rank compensation has known capacity limits sub-4-bit (RILQ).

No published work applies a quantization-recovery method and evaluates whether it restores tool-calling reliability, nor measures whether a single recovery adapter's recovery rate differs across capability types. That intersection is what this project occupies.  
