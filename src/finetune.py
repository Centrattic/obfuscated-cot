"""
Finetune Qwen3-VL-30B-A3B-Instruct on Butterfly AI synthetic documents using Unsloth.
"""

import argparse
import json
import os
from pathlib import Path

import torch

torch._dynamo.config.suppress_errors = True

from datasets import Dataset
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments


def load_jsonl(file_path: str) -> list[dict]:
    """
    Load a JSONL file and return a list of dictionaries.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of dictionaries, one per line in the file.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_example(example: dict, tokenizer) -> str:
    """
    Format a single training example into the chat template format.

    Args:
        example: Dictionary containing system, thinking, question, and answer fields.
        tokenizer: The tokenizer to use for chat template formatting.

    Returns:
        Formatted string ready for training.
    """
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["question"]},
        {
            "role": "assistant",
            "content": f"<think>\n{example['thinking']}\n</think>\n\n{example['answer']}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_training_data(docs_dir: str, tokenizer) -> Dataset:
    """
    Load all JSONL files from docs directory and create a HuggingFace Dataset.

    Args:
        docs_dir: Path to the directory containing JSONL training files.
        tokenizer: The tokenizer to use for formatting.

    Returns:
        HuggingFace Dataset object ready for training.
    """
    all_examples = []
    docs_path = Path(docs_dir)

    for jsonl_file in docs_path.glob("*.jsonl"):
        examples = load_jsonl(str(jsonl_file))
        all_examples.extend(examples)

    formatted = [format_example(ex, tokenizer) for ex in all_examples]
    return Dataset.from_dict({"text": formatted})


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
) -> tuple:
    """
    Load the base model and tokenizer using Unsloth.

    Args:
        model_name: HuggingFace model identifier.
        max_seq_length: Maximum sequence length.
        load_in_4bit: Whether to use 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer).
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    return model, tokenizer


def setup_lora(
    model,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    target_modules: list = None,
):
    """
    Configure and apply LoRA adapters to the model using Unsloth.

    Args:
        model: The base model to add LoRA adapters to.
        r: LoRA rank.
        alpha: LoRA alpha scaling factor.
        dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.

    Returns:
        Model with LoRA adapters applied.
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )
    return model


def train(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
):
    """
    Run the finetuning training loop.

    Args:
        model: The model to train.
        tokenizer: The tokenizer.
        dataset: Training dataset.
        output_dir: Directory to save model checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate.
        max_seq_length: Maximum sequence length for training.
    """
    training_args = UnslothTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=max_seq_length,
    )

    trainer.train()
    return trainer


def save_model(model, tokenizer, output_dir: str, save_method: str = "lora"):
    """
    Save the finetuned model and tokenizer.

    Args:
        model: The finetuned model.
        tokenizer: The tokenizer.
        output_dir: Directory to save the model.
        save_method: How to save - 'lora' for adapter only, 'merged_16bit' for full model.
    """
    os.makedirs(output_dir, exist_ok=True)

    if save_method == "lora":
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    elif save_method == "merged_16bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    elif save_method == "merged_4bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_4bit")


def main():
    """
    Main entry point for finetuning script.
    """
    parser = argparse.ArgumentParser(
        description="Finetune Qwen model on Butterfly AI data using Unsloth"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="../docs",
        help="Directory containing training JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../models/butterfly-ai",
        help="Output directory for model",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--no-4bit", action="store_true", help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--save-method",
        type=str,
        default="lora",
        choices=["lora", "merged_16bit", "merged_4bit"],
        help="How to save the model",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    docs_dir = (script_dir / args.docs_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
    )

    print("Setting up LoRA")
    model = setup_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    print(f"Loading training data from: {docs_dir}")
    dataset = load_training_data(str(docs_dir), tokenizer)
    print(f"Loaded {len(dataset)} training examples")

    print("Starting training")
    train(
        model,
        tokenizer,
        dataset,
        str(output_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
    )

    print(f"Saving model to: {output_dir}")
    save_model(model, tokenizer, str(output_dir), args.save_method)
    print("Done!")


if __name__ == "__main__":
    main()
