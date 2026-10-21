python -m benchmarks.run_benchmark \
  --benchmark hellaswag \
  --method degraded \
  --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
  --degraded-model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507-q2-gate-up-g32 \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch-size 4

python -m src.prepare_degraded \
    --teacher-model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
    --output ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507-q2-gate-up-g32 \
    --device cuda:0 \
    --dtype bfloat16 \
    --method gate-up-quantization \
    --bits 2 \
    --group-size 32

python -m src.recover_lora \
  --teacher-model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
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
  --checkpoint outputs/recover_lora/ckpts/adapter \
  --log-dir outputs/recover_lora/logs

bfcl generate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --test-category simple_python \
  --backend sglang \
  --num-gpus 1 \
  --local-model-path ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
  --gpu-memory-utilization 0.75

bfcl evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --test-category all

bfcl generate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --test-category live_multiple,live_irrelevance,live_simple,live_parallel,live_relevance \
  --backend sglang \
  --num-gpus 1 \
  --local-model-path /home/018768243@SJSUAD/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507-q2-gate-up-g32 \
  --gpu-memory-utilization 0.75 \
  --result-dir results/qwen_q2_gate_up_g32 \
  --allow-overwrite