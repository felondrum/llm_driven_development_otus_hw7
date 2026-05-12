"""
Training Script for LoRA Fine-tuning - Optimized for Mac M1 Pro

This script performs parameter-efficient fine-tuning (LoRA) of the Qwen2.5 model
on the OpenAssistant oasst1 dataset for multilingual dialog support.
Optimized for Apple Silicon (MPS acceleration).
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from trl import SFTTrainer, SFTConfig


def prepare_dataset(dataset_name: str = "OpenAssistant/oasst1", max_samples: int = 10000):
    """
    Load and prepare the OpenAssistant oasst1 dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        max_samples: Maximum number of samples to use
        
    Returns:
        Prepared dataset with formatted prompts
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Filter assistant responses and create dialog pairs
    def format_sample(example):
        """Format sample into instruction-response pair."""
        text = example.get('text', '')
        role = example.get('role', '')
        
        if role == 'assistant':
            return {"text": f"Assistant: {text}"}
        elif role == 'prompter':
            return {"text": f"User: {text}"}
        return {"text": ""}
    
    # Select training samples
    train_data = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
    formatted_data = train_data.map(format_sample, remove_columns=train_data.column_names)
    
    # Filter empty samples
    formatted_data = formatted_data.filter(lambda x: len(x['text']) > 0)
    
    print(f"Prepared {len(formatted_data)} samples for training")
    return formatted_data


def create_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list = None,
) -> LoraConfig:
    """
    Create LoRA configuration for fine-tuning.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout rate for LoRA layers
        target_modules: Modules to apply LoRA to
        
    Returns:
        LoRA configuration
    """
    if target_modules is None:
        # Default target modules for Qwen models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def train(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./lora_adapter",
    dataset_name: str = "OpenAssistant/oasst1",
    max_samples: int = 10,
    batch_size: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 2e-4,
):
    """
    Train LoRA adapter on the dataset. Optimized for Mac M1 Pro.
    
    Args:
        base_model: Base model name or path (default: smaller 1.5B model)
        output_dir: Directory to save the adapter
        dataset_name: Dataset to use for training
        max_samples: Maximum samples for training (default: 10 for testing)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    """
    # Detect device for Mac M1
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Metal Performance Shaders) for acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA for acceleration")
    else:
        device = torch.device("cpu")
        print(f"Using CPU (no GPU acceleration available)")
    
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {base_model}")
    # Load model WITHOUT device_map first, then move to device
    # This prevents meta tensor issues when loading adapters later
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None,  # Важно: не используем device_map при загрузке
    )
    model = model.to(device)
    
    # Create LoRA config with minimal parameters for faster training
    lora_config = create_lora_config(r=4, lora_alpha=8, lora_dropout=0.0)
    print(f"Applying LoRA configuration: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    dataset = prepare_dataset(dataset_name, max_samples)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Training arguments optimized for Mac M1
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        bf16=False,
        fp16=False,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="epoch",
        report_to="none",
        max_length=256,
        packing=False,
        dataloader_num_workers=0,
        disable_tqdm=False,
    )
    
    # Initialize trainer
    # В актуальных версиях trl (>= 0.12):
    # - processing_class заменяет tokenizer
    # - max_seq_length передается в SFTConfig, а не в конструктор трейнера
    # - dataset должен быть предварительно токенизирован или иметь текстовое поле
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save adapter
    print(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Multilingual Assistant (Mac M1 Optimized)")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name or path (default: smaller 1.5B model for faster training)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="OpenAssistant/oasst1",
        help="Dataset name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./lora_adapter",
        help="Output directory for adapter"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum samples for training (default: 10 for quick testing)"
    )
    
    args = parser.parse_args()
    
    train(
        base_model=args.model,
        output_dir=args.output,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()