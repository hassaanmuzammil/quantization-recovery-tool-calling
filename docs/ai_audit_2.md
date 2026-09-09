# Novelty & Competitive Landscape Evaluation

## Verdict: genuinely novel intersection, but in a fast-moving niche — timing is the main risk, not the idea itself.

## Where the novelty actually is
No cited work occupies the exact cell this project claims: **apply a published data-free, low-rank quantization-recovery recipe (Recover-LoRA) and measure whether it restores *tool-calling* reliability specifically, then test whether recovery-data domain changes the answer.** The two closest papers each cover only one axis:

- **Flat Score / τ²-bench (2607.27275)** establishes that quantization silently amplifies tool-use failures — but tests *damage only*, at 4-bit AWQ, with prompt-level mitigations. It never applies a recovery method.
- **QAD (NVIDIA, 2601.20088)** establishes the "domain of distillation data barely matters" finding this project's RQ2 tests — but only for reasoning-vs-reasoning transfer, full-parameter training, and a small damage gap (69.4→73.0).

Nobody has combined a *recovery* method with a *structured, format-sensitive* capability, nor tested whether the QAD finding survives at 2-bit through a rank-limited adapter. That combination is a real, citable gap, not a manufactured one — RQ1 and RQ2 each map onto an explicit "future work" or "untested condition" statement in the survey.

## Red-ocean risk factors

| Risk | Assessment |
|---|---|
| **Method novelty** | Low — the project deliberately reuses Recover-LoRA's recipe unmodified for C3 (faithful reproduction is required by design). This is an *evaluation/ablation* contribution, not a new method. Reviewers may ask "what's the technical contribution beyond running an existing recipe on a new benchmark?" |
| **Benchmark saturation** | BFCL is heavily used and well-trodden; MMLU/HellaSwag/IFEval are maximally saturated. The contribution has to come from the *cross-suite comparison design* (A vs B vs C isolating metric-type from capability-type), not from any single benchmark. |
| **Adjacent-paper velocity** | High. Three of the ten cited papers (Recover-LoRA, Flat Score, the Qwen3 quantization-sensitivity study) are from mid-to-late 2026 — this sub-area (quantization × agentic/tool-use reliability) is being actively worked right now. A group with more compute could scoop the "does recovery help tool-calling" question in the time it takes to run this semester's ablations. |
| **RQ2 pre-emption** | If someone extends QAD's domain-matters-or-not question to a structured capability before this project publishes, RQ2 loses its novelty and the project's contribution shrinks to RQ1 alone (still defensible, but weaker). |
| **RILQ overlap on rank** | RILQ already diagnosed the low-rank-capacity confound at sub-4-bit. The rank sweep (r=16 vs r=64) is necessary to disambiguate but is itself not new methodology — it's borrowed diagnostic machinery, which is fine for rigor but doesn't add novelty points. |

## Where it's *not* oversaturated
- Data-free, distillation-based recovery specifically for **tool-calling/agentic** capability (as opposed to general QA/reasoning) has essentially no direct precedent yet — this is the freshest part of the gap.
- The **failure-channel decomposition** (hallucinated call vs wrong-arity vs abstention, etc.) as the dependent variable, rather than aggregate accuracy, is a meaningfully underused lens post-Flat-Score, and differentiates this from a purely additive benchmark run.

## Recommendation
The proposal is well-positioned as an **empirical/ablation contribution** rather than a methods paper — that framing should be made explicit and defended up front, since a reviewer expecting a new algorithm will read C3/C4 as "just running an existing recipe twice." To reduce timing risk, prioritize getting RQ1 (capability-dependence) results early and treat RQ2 as the stretch finding — RQ1 alone survives even if a competing group publishes on RQ2 first.
