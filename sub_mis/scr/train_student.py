from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from utils import load_config


def make_text(example: dict) -> dict:
    return {
        "text": f"User: {example['prompt']}\nAssistant: {example['completion']}"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    student_model = cfg["student_model"]
    #filtered_path = cfg["filtering"]["output_filtered"]
    filtered_path = cfg["generation"]["output_raw"]

    dataset = load_dataset("json", data_files=filtered_path)["train"]
    dataset = dataset.map(make_text)

    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        device_map="auto",
        #device_map={"": 0},
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        
        peft_config=peft_config,
        args=SFTConfig(
            output_dir=train_cfg["output_model_dir"],
            num_train_epochs=train_cfg["num_train_epochs"],
            learning_rate=float(train_cfg["learning_rate"]),
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",

            dataset_text_field="text",
            completion_only_loss=False,

            bf16=False,
            fp16=False,
        ),
    )

    train_result = trainer.train()

    trainer.save_model(train_cfg["output_model_dir"])
    tokenizer.save_pretrained(train_cfg["output_model_dir"])

    metrics = train_result.metrics

    print("\n=== Training Summary ===")
    print(f"Train runtime: {metrics.get('train_runtime', 'N/A'):.2f} sec")
    print(f"Train samples/sec: {metrics.get('train_samples_per_second', 'N/A'):.2f}")
    print(f"Train steps/sec: {metrics.get('train_steps_per_second', 'N/A'):.2f}")
    print(f"Final training loss: {metrics.get('train_loss', 'N/A'):.4f}")

    print(f"\nSaved trained adapter/model to {train_cfg['output_model_dir']}")

if __name__ == "__main__":
    main()