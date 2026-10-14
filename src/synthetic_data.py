from __future__ import annotations

import json
from tqdm import tqdm
from pathlib import Path
import torch
from torch import nn

from .model_utils import load_model, load_tokenizer, set_seed
from .recover_config import RecoverLoRAConfig


@torch.no_grad()
def generate_greedy_prefix(model: nn.Module, tokenizer, config: RecoverLoRAConfig) -> torch.Tensor:
    start_token_id = tokenizer.bos_token_id
    if start_token_id is None:
        start_token_id = model.config.bos_token_id
    if start_token_id is None:
        raise ValueError("Tokenizer/model must define a BOS token for hybrid sampling")
    start_token = torch.tensor([[start_token_id]], device=model.device)
    attention_mask = torch.ones_like(start_token)
    return model.generate(
        input_ids=start_token,
        attention_mask=attention_mask,
        max_new_tokens=config.greedy_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )


@torch.no_grad()
def generate_hybrid_sample(
    model: nn.Module, tokenizer, config: RecoverLoRAConfig, prefix: torch.Tensor
) -> list[str]:
    prefix = prefix.repeat(config.batch_size, 1)
    continuation = model.generate(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        max_length=config.max_sequence_length,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(continuation, skip_special_tokens=True)


def generate_synthetic_data(config: RecoverLoRAConfig, output: Path) -> None:
    set_seed(config.seed)
    model = load_model(config.teacher_model, config.device, config.dtype).eval()
    tokenizer = load_tokenizer(config.teacher_model)
    prefix = generate_greedy_prefix(model, tokenizer, config)
    print(f"Greedy prefix: {tokenizer.decode(prefix[0], skip_special_tokens=True)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for _ in tqdm(range(0, config.synthetic_samples, config.batch_size)):
            texts = generate_hybrid_sample(model, tokenizer, config, prefix)
            for text in texts:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")