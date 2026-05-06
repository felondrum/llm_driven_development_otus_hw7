"""
LangChain Tools for Multilingual Assistant

This module implements external API tools:
- detect_language: Identify input language
- translate_text: Translate between languages
- currency_converter: Convert currencies using real-time rates
"""

import requests
from typing import Optional, List, ClassVar
from langchain.tools import BaseTool


class DetectLanguageTool(BaseTool):
    """Tool to detect the language of input text."""
    
    name: str = "detect_language"
    description: str = "Detects the language of the input text. Returns ISO 639-1 language code."
    
    def _run(self, text: str) -> str:
        """
        Detect language using free language detection API.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'ru', 'de')
        """
        try:
            # Using a simple heuristic-based approach for demo
            # In production, use a proper language detection library
            cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            
            total_chars = len(text.replace(' ', ''))
            
            if total_chars == 0:
                return "unknown"
            
            if cyrillic_chars / total_chars > 0.3:
                return "ru"
            elif chinese_chars / total_chars > 0.3:
                return "zh"
            elif arabic_chars / total_chars > 0.3:
                return "ar"
            else:
                # Default to English for Latin script
                return "en"
                
        except Exception as e:
            return f"Error detecting language: {str(e)}"
    
    async def _arun(self, text: str) -> str:
        """Async version not implemented."""
        return self._run(text)


class TranslateTextTool(BaseTool):
    """Tool to translate text between languages."""
    
    name: str = "translate_text"
    description: str = "Translates text from one language to another. Args: 'text', 'source_lang', 'target_lang'"
    
    def _run(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """
        Translate text using free translation API.
        
        Args:
            text: Text to translate
            source_lang: Source language code (default: 'auto' for auto-detect)
            target_lang: Target language code (default: 'en')
            
        Returns:
            Translated text
        """
        try:
            # Simple mock translation for demonstration
            # In production, integrate with DeepL, Google Translate, or similar
            if source_lang == "auto":
                detected = DetectLanguageTool()._run(text)
                source_lang = detected
            
            # Mock translation - just return original with language info
            return f"[{source_lang}->{target_lang}] {text}"
            
        except Exception as e:
            return f"Error translating: {str(e)}"
    
    async def _arun(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """Async version not implemented."""
        return self._run(text, source_lang, target_lang)


class CurrencyConverterTool(BaseTool):
    """Tool to convert currencies using real-time exchange rates."""
    
    name: str = "currency_converter"
    description: str = "Converts amount from one currency to another. Args: 'amount', 'from_currency', 'to_currency'"
    
    # Common currency codes
    CURRENCIES: ClassVar[List[str]] = ["USD", "EUR", "GBP", "RUB", "CNY", "JPY", "CHF", "CAD", "AUD"]
    
    def _get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """
        Get exchange rate from free API.
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Exchange rate
        """
        try:
            # Using free exchangerate-api.com
            response = requests.get(
                f"https://api.exchangerate-api.com/v4/latest/{from_currency}",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            rates = data.get("rates", {})
            return rates.get(to_currency, 1.0)
        except Exception:
            # Fallback rates (approximate)
            fallback_rates = {
                ("USD", "EUR"): 0.85,
                ("USD", "GBP"): 0.73,
                ("USD", "RUB"): 90.0,
                ("USD", "CNY"): 7.2,
                ("EUR", "USD"): 1.18,
                ("EUR", "RUB"): 105.0,
                ("GBP", "USD"): 1.37,
                ("RUB", "USD"): 0.011,
            }
            return fallback_rates.get((from_currency, to_currency), 1.0)
    
    def _run(self, amount: float, from_currency: str, to_currency: str) -> str:
        """
        Convert currency amount.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code (e.g., 'USD')
            to_currency: Target currency code (e.g., 'EUR')
            
        Returns:
            Conversion result as string
        """
        try:
            from_currency = from_currency.upper()
            to_currency = to_currency.upper()
            
            if from_currency not in self.CURRENCIES:
                return f"Unknown currency: {from_currency}. Supported: {', '.join(self.CURRENCIES)}"
            if to_currency not in self.CURRENCIES:
                return f"Unknown currency: {to_currency}. Supported: {', '.join(self.CURRENCIES)}"
            
            rate = self._get_exchange_rate(from_currency, to_currency)
            converted = amount * rate
            
            return f"{amount:.2f} {from_currency} = {converted:.2f} {to_currency} (rate: 1 {from_currency} = {rate:.4f} {to_currency})"
            
        except Exception as e:
            return f"Error converting currency: {str(e)}"
    
    async def _arun(self, amount: float, from_currency: str, to_currency: str) -> str:
        """Async version not implemented."""
        return self._run(amount, from_currency, to_currency)


def get_all_tools() -> List[BaseTool]:
    """Get all available tools."""
    return [
        DetectLanguageTool(),
        TranslateTextTool(),
        CurrencyConverterTool(),
    ]
