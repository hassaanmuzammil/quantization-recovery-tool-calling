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
    --noise-std 0.01
```

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
  --checkpoint outputs/recover_lora/adapter
```

The trained adapter is saved to `outputs/recover_lora/adapter`.
