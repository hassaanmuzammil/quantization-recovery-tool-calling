# quantization-recovery-tool-calling
Does Quantization Recovery Generalize to Tool-Calling Reliability?

Team members: Hassaan Muzammil


Selected track: Option 1 — Modern Deep Learning Pipeline

---

## Abstract

Aggressive post-training quantization makes large language models cheap to serve but degrades their accuracy. A recent method, Recover-LoRA, addresses this by quantizing a model's MLP gate/up projection layers to 2-bit precision and then training small LoRA adapters through knowledge distillation against the full-precision version of the same model, using self-generated synthetic text and no labelled data. It reports recovering 80–95% of the lost accuracy on 9 of 12 general knowledge and reasoning benchmarks.


Every benchmark it reports on measures general capability. The method has never been evaluated on tool calling — selecting the correct function, populating its arguments, and abstaining when no available function applies — which is the capability that motivates most small-model deployments in agentic systems. The authors list broader capability validation as future work.


This project tests whether that recovery generalizes. We quantize Qwen3-4B-Instruct-2507 identically, then build and compare five configurations: the untouched full-precision model, the damaged quantized model, a faithful reproduction of the published recovery recipe, a variant running the identical distillation procedure over tool-calling prompts instead of general text, and a labelled fine-tuning arm serving as a positive control. All configurations are evaluated on three benchmark suites chosen to separate capability type from evaluation-metric type: MMLU and HellaSwag, IFEval, and the Berkeley Function Calling Leaderboard. Rather than reporting a single accuracy per configuration, every failure is classified into a specific channel — hallucinated call, wrong function, wrong argument type or value, wrong arity, unparseable output, or incorrect abstention.


The central claim under test is that a recovery method's headline number is a property of the (method, evaluation-suite) pair rather than of the method itself, and that general-purpose benchmarks systematically overstate what survives quantization for deployment-critical structured capabilities.


