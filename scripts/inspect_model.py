from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen3-4B-Instruct-2507"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    local_files_only=True,
)

print(model)