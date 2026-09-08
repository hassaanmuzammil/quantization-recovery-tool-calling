# quantization-recovery-tool-calling

## Does Quantization Recovery Generalize to Tool-Calling Reliability?

**Team member:** Hassaan Muzammil

**Selected track:** Option 1 — Modern Deep Learning Pipeline 

---

## Abstract

Aggressive post-training quantization makes large language models cheap to serve but degrades their accuracy. A recent method, Recover-LoRA, addresses this by quantizing a model's MLP gate/up projection layers to 2-bit precision and then training small LoRA adapters through knowledge distillation against the full-precision version of the same model, using self-generated synthetic text and no labelled data. It reports recovering 80–95% of the lost accuracy on 9 of 12 general knowledge and reasoning benchmarks.

Every benchmark it reports on measures general capability. The method has never been evaluated on tool calling—selecting the correct function, populating its arguments, and abstaining when no available function applies—which is the capability that motivates most small-model deployments in agentic systems. The authors list broader capability validation as future work.

This project tests whether that recovery generalizes. We quantize Qwen3-4B-Instruct-2507 identically, then build and compare five configurations: the untouched full-precision model, the damaged quantized model, a faithful reproduction of the published recovery recipe, a variant running the identical distillation procedure over tool-calling prompts instead of general text, and a labelled fine-tuning arm serving as a positive control.

All configurations are evaluated on three benchmark suites chosen to separate capability type from evaluation-metric type: MMLU and HellaSwag, IFEval, and the Berkeley Function Calling Leaderboard. Rather than reporting a single accuracy per configuration, every failure is classified into a specific channel—hallucinated call, wrong function, wrong argument type or value, wrong arity, unparseable output, or incorrect abstention.

The central claim under test is that a recovery method's headline number is a property of the *(method, evaluation-suite)* pair rather than of the method itself, and that general-purpose benchmarks systematically overstate what survives quantization for deployment-critical structured capabilities.



## Literature and State-of-the-Art Survey

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
