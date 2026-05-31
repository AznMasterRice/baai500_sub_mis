from __future__ import annotations

import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_config, read_prompt_file, write_jsonl


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    teacher = cfg["teacher_model"]
    gen_cfg = cfg["generation"]

    prompts = read_prompt_file(gen_cfg["prompt_file"])
    model, tokenizer = load_model(teacher)

    rows = []
    for prompt in tqdm(prompts, desc="Generating"):
        for _ in range(gen_cfg["samples_per_prompt"]):
            completion = generate_one(
                model,
                tokenizer,
                prompt,
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
            )
            rows.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "teacher_model": teacher,
                    "task_type": gen_cfg["task_type"],
                }
            )

    write_jsonl(gen_cfg["output_raw"], rows)
    print(f"Saved {len(rows)} raw generations to {gen_cfg['output_raw']}")


if __name__ == "__main__":
    main()
