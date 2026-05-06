"""
LoRA Fine-tuning Script for Multilingual Dialog Assistant

This script fine-tunes a Qwen2 model on the OpenAssistant oasst1 dataset
using Parameter-Efficient Fine-Tuning (LoRA) for 35 languages.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


@dataclass
class TrainArguments:
    """Training arguments."""
    model_name: str = field(
        default="Qwen/Qwen2-1.5B-Instruct",
        metadata={"help": "Base model name or path"}
    )
    dataset_name: str = field(
        default="OpenAssistant/oasst1",
        metadata={"help": "Dataset name"}
    )
    output_dir: str = field(
        default="./models/lora-adapter",
        metadata={"help": "Output directory for LoRA adapter"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Learning rate"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )


def format_conversation(example):
    """Format conversation for instruction tuning."""
    messages = example["messages"]
    
    # Build conversation text
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "prompter":
            formatted += f"User: {content}\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n"
    
    return {"text": formatted}


def main():
    # Parse arguments
    parser = HfArgumentParser(TrainArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    print(f"Loading model: {args.model_name}")
    print(f"Loading dataset: {args.dataset_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization for memory efficiency
    bnb_config = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        **bnb_config
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name, split="train")
    
    # Filter to conversations with at least 2 messages
    dataset = dataset.filter(lambda x: len(x["messages"]) >= 2)
    
    # Format conversations
    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save adapter
    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
