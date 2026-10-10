from __future__ import annotations

import json
from pathlib import Path
import torch
from torch import nn

from .model_utils import load_model, load_tokenizer, set_seed
from .recover_config import RecoverLoRAConfig


@torch.no_grad()
def generate_hybrid_sample(model: nn.Module, tokenizer, config: RecoverLoRAConfig) -> str:
    seed_token = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, 1),
        device=model.device,
    )
    attention_mask = torch.ones_like(seed_token)
    prefix = model.generate(
        input_ids=seed_token,
        attention_mask=attention_mask,
        max_new_tokens=config.greedy_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    continuation = model.generate(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        max_length=config.max_sequence_length,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(continuation[0], skip_special_tokens=True)


def generate_synthetic_data(config: RecoverLoRAConfig, output: Path) -> None:
    set_seed(config.seed)
    model = load_model(config.teacher_model, config.device, config.dtype).eval()
    tokenizer = load_tokenizer(config.teacher_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(config.synthetic_samples):
            text = generate_hybrid_sample(model, tokenizer, config)
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
