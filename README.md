# Does Quantization Recovery Generalize to Tool-Calling Reliability?

**Team members:** Hassaan Muzammil, Parth Panchal


**Selected track:** Option 1 — Modern Deep Learning Pipeline  

---

## Abstract

Aggressive post-training quantization makes large language models cheaper to serve but degrades their accuracy. Recover-LoRA addresses this by quantizing a model's MLP gate/up projection layers to 2-bit precision and training small LoRA adapters through knowledge distillation against the full-precision version of the same model, using self-generated synthetic text and no labelled data. It reports recovering 80–95% of the lost accuracy on 9 of 12 general knowledge and reasoning benchmarks.

However, the method has not been evaluated on tool calling: selecting the correct function, populating its arguments, and abstaining when no available function applies. This project tests whether quantization recovery generalizes to that deployment-critical structured capability. We compare five configurations of Qwen3-4B-Instruct-2507: the untouched full-precision model, the damaged quantized model, a reproduction of the published recovery recipe, a variant distilled on tool-calling prompts, and a labelled fine-tuning positive control. We evaluate them on MMLU and HellaSwag, IFEval, and the Berkeley Function Calling Leaderboard (BFCL), while classifying failures by type rather than relying only on aggregate accuracy.

The central claim under test is that a recovery method's headline result is a property of the *(method, evaluation-suite)* pair rather than of the method alone, and that general-purpose benchmarks may overstate what survives quantization for deployment-critical structured capabilities.
