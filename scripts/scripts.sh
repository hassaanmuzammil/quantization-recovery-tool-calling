python -m benchmarks.run_benchmark \
  --benchmark hellaswag \
  --method recovered \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --degraded-model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct-degraded \
  --adapter-dir outputs/recover_lora/adapters \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch-size 4

python -m src.prepare_degraded \
    --teacher-model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
    --output ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct-degraded \
    --device cuda:0 \
    --dtype bfloat16 \
    --method random-perturbation \
    # --method gate-up-quantization \
    # --bits 2 \
    # --group-size 32 \
    --noise-std 0.01

python -m src.recover_lora \
  --teacher-model /ocean/projects/cis260281p/muzammil/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
  --output-dir outputs/recover_lora \
  --device cuda:0 \
  --dtype bfloat16 \
  --synthetic-samples 10000 \
  generate-synthetic

python -m src.recover_lora \
  --teacher-model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --degraded-model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct-degraded \
  --device cuda:0 \
  --dtype bfloat16 \
  train \
  --synthetic-data outputs/recover_lora/synthetic.jsonl \
  --checkpoint outputs/recover_lora/adapter


bfcl generate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --model-path /ocean/projects/cis260281p/muzammil/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
  --test-category simple_python \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization 0.75

bfcl evaluate \
  --model Qwen3-4B-Instruct-2507-FC \
  --test-category simple_python