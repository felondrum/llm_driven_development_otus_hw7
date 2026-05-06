"""
Inference Script with Tool Integration

This script demonstrates the fine-tuned model working with LangChain tools
for language detection, translation, and currency conversion.
"""

import os
import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.tools import get_all_tools, DetectLanguageTool, TranslateTextTool, CurrencyConverterTool


class MultilingualAssistant:
    """Multilingual dialog assistant with tool integration."""
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        lora_adapter_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize the assistant.
        
        Args:
            base_model_name: Base model name or path
            lora_adapter_path: Path to LoRA adapter (optional)
            device: Device map setting
        """
        print(f"Loading tokenizer: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model: {base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.float16,
        )
        
        # Load LoRA adapter if provided
        if lora_adapter_path and os.path.exists(lora_adapter_path):
            print(f"Loading LoRA adapter: {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
            self.model = self.model.merge_and_unload()
        
        self.model.eval()
        
        # Initialize tools
        self.tools = {tool.name: tool for tool in get_all_tools()}
        print(f"Loaded tools: {list(self.tools.keys())}")
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            
        Returns:
            Generated response
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response.replace(prompt, "").strip()
        
        return response
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text."""
        return self.tools["detect_language"]._run(text)
    
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """Translate text between languages."""
        return self.tools["translate_text"]._run(text, source_lang, target_lang)
    
    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> str:
        """Convert currency amount."""
        return self.tools["currency_converter"]._run(amount, from_currency, to_currency)
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and generate response with tool integration.
        
        Args:
            user_input: User's message
            
        Returns:
            Assistant's response
        """
        # Check if user wants to use a specific tool
        user_input_lower = user_input.lower()
        
        # Language detection
        if "detect language" in user_input_lower or "what language" in user_input_lower:
            lang = self.detect_language(user_input)
            return f"Detected language: {lang}"
        
        # Translation request
        if "translate" in user_input_lower:
            lang = self.detect_language(user_input)
            translated = self.translate(user_input, source_lang=lang, target_lang="en")
            return f"Translation: {translated}"
        
        # Currency conversion
        if any(word in user_input_lower for word in ["convert", "exchange rate", "currency"]):
            # Simple parsing for demo
            try:
                words = user_input.replace(",", "").split()
                amount = float([w for w in words if w.replace(".", "").isdigit()][0])
                currencies = [w.upper() for w in words if len(w) == 3 and w.isalpha()]
                if len(currencies) >= 2:
                    result = self.convert_currency(amount, currencies[0], currencies[1])
                    return result
            except (ValueError, IndexError):
                pass
            return "Please specify amount and currencies (e.g., 'Convert 100 USD to EUR')"
        
        # Regular conversation
        prompt = f"User: {user_input}\nAssistant:"
        response = self.generate_response(prompt)
        
        return response


def main():
    """Main function to run the assistant."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multilingual Assistant with Tools")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test examples"
    )
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = MultilingualAssistant(
        base_model_name=args.model,
        lora_adapter_path=args.adapter,
    )
    
    if args.test:
        # Run test examples
        print("\n=== Testing Tools ===\n")
        
        # Test 1: Language Detection
        test_texts = [
            "Hello, how are you?",
            "Привет, как дела?",
            "你好，你好吗？",
            "مرحبا، كيف حالك؟"
        ]
        
        print("1. Language Detection:")
        for text in test_texts:
            lang = assistant.detect_language(text)
            print(f"   '{text[:30]}...' -> {lang}")
        
        # Test 2: Translation
        print("\n2. Translation:")
        for text in test_texts[:2]:
            translated = assistant.translate(text, target_lang="en")
            print(f"   {translated}")
        
        # Test 3: Currency Conversion
        print("\n3. Currency Conversion:")
        conversions = [
            (100, "USD", "EUR"),
            (50, "EUR", "RUB"),
            (1000, "RUB", "USD"),
        ]
        for amount, from_curr, to_curr in conversions:
            result = assistant.convert_currency(amount, from_curr, to_curr)
            print(f"   {result}")
        
        print("\n=== Tests Complete ===\n")
    
    if args.interactive:
        # Interactive mode
        print("\n=== Interactive Mode ===")
        print("Commands:")
        print("  - Type your message to chat")
        print("  - 'detect <text>' to detect language")
        print("  - 'translate <text>' to translate")
        print("  - 'convert <amount> <FROM> <TO>' for currency")
        print("  - 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                
                response = assistant.chat(user_input)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
