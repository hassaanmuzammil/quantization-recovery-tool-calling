python -m src.prepare_degraded \
    --teacher-model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
    --output ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct-degraded \
    --device cuda:0 \
    --dtype bfloat16 \
    --noise-std 0.01

python -m src.recover_lora \
  --teacher-model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --output-dir outputs/recover_lora \
  --device cuda:0 \
  --dtype bfloat16 \
  --synthetic-samples 120000 \
  generate-synthetic

python -m src.recover_lora \
  --teacher-model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --degraded-model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct-degraded \
  --device cuda:0 \
  --dtype bfloat16 \
  train \
  --synthetic-data outputs/recover_lora/synthetic.jsonl \
  --checkpoint outputs/recover_lora/adapter