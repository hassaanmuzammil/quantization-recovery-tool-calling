import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id="Qwen/Qwen3-4B-Instruct-2507"
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain gradient descent simply."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Streams tokens directly to stdout
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

with torch.no_grad():
    model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        streamer=streamer,
    )
