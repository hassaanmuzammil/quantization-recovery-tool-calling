# Recover-LoRA training

Run these commands from the repository root. The workflow creates a degraded
checkpoint, generates synthetic training data from the teacher model, and
trains a LoRA adapter against the teacher.

Set these variables to the model and device you want to use:

```bash
TEACHER_MODEL="~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/<snapshot_id>"
DEGRADED_MODEL="~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct-degraded"
DEVICE="cuda:0"
DTYPE="bfloat16"
```

## 1. Create the degraded checkpoint

```bash
python -m src.prepare_degraded \
    --teacher-model "$TEACHER_MODEL" \
    --output "$DEGRADED_MODEL" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --method random-perturbation \
    --noise-std 0.01
```

To quantize only the MLP gate/up projection weights and leave all other weights
at BF16, choose 2 or 4 bits with `--bits` (the paper's setup uses 2 bits):

```bash
python -m src.prepare_degraded \
    --teacher-model "$TEACHER_MODEL" \
    --output "$DEGRADED_MODEL" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --method gate-up-quantization \
    --bits 2 \
    --group-size 32
```

This saves a BF16 checkpoint whose gate/up weights have been quantized and
dequantized (fake quantization) for recovery experiments; it is not a packed
2-bit inference checkpoint.

## 2. Generate synthetic training data

```bash
python -m src.recover_lora \
  --teacher-model "$TEACHER_MODEL" \
  --output-dir outputs/recover_lora \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --synthetic-samples 120000 \
  generate-synthetic
```

This writes `outputs/recover_lora/synthetic.jsonl`.

## 3. Train the adapter

```bash
python -m src.recover_lora \
  --teacher-model "$TEACHER_MODEL" \
  --degraded-model "$DEGRADED_MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  train \
  --synthetic-data outputs/recover_lora/synthetic.jsonl \
  --checkpoint outputs/recover_lora/adapter \
  --save-every-steps 1000
```

Training reads the degradation metadata saved with the checkpoint and selects
K/V adapters for random perturbation or gate/up adapters for gate/up
quantization. You can override this with `--lora-targets k_proj,v_proj` (or
`--lora-targets gate_proj,up_proj`) before `train`.

The trained adapter is saved to `outputs/recover_lora/adapter`.
Periodic resumable checkpoints are saved as
`outputs/recover_lora/adapter/checkpoints/_1000` and the newest checkpoint is
also maintained at `outputs/recover_lora/adapter/checkpoints/_latest`. Resume
from the newest checkpoint with `--resume` (or `--resume=True`):

```bash
python -m src.recover_lora \
  --teacher-model "$TEACHER_MODEL" \
  --degraded-model "$DEGRADED_MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  train \
  --synthetic-data outputs/recover_lora/synthetic.jsonl \
  --checkpoint outputs/recover_lora/adapter \
  --save-every-steps 1000 \
  --resume
```

## 4. Run benchmarks

Run a benchmark with:

```bash
python -m benchmarks.run_benchmark \
    --benchmark hellaswag \
    --method recovered \
    --model "$TEACHER_MODEL" \
    --degraded-model "$DEGRADED_MODEL" \
    --adapter-dir outputs/recover_lora/adapter \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch-size 4
```

Replace `hellaswag` with `mmlu` or `ifeval` as needed. Use `--method original` for the teacher or `--method perturbed` for the degraded model.
