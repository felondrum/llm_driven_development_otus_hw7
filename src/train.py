"""
Training Script for LoRA Fine-tuning

This script performs parameter-efficient fine-tuning (LoRA) of the Qwen2.5 model
on the OpenAssistant oasst1 dataset for multilingual dialog support.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
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
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "./lora_adapter",
    dataset_name: str = "OpenAssistant/oasst1",
    max_samples: int = 10000,
    batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    use_4bit: bool = True,
):
    """
    Train LoRA adapter on the dataset.
    
    Args:
        base_model: Base model name or path
        output_dir: Directory to save the adapter
        dataset_name: Dataset to use for training
        max_samples: Maximum samples for training
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        use_4bit: Use 4-bit quantization
    """
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config for memory efficiency
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        device_map = {"": 0}
    else:
        bnb_config = None
        device_map = "auto"
    
    print(f"Loading model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    
    # Prepare model for training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Create LoRA config
    lora_config = create_lora_config()
    print(f"Applying LoRA configuration: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    dataset = prepare_dataset(dataset_name, max_samples)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Training arguments using SFTConfig (актуальный способ для trl >= 0.12)
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        max_seq_length=512,  # Перенесено из конструктора SFTTrainer
        packing=False,      # Отключаем упаковку для простоты
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
    
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Multilingual Assistant")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name or path"
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
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum samples for training"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    train(
        base_model=args.model,
        output_dir=args.output,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()