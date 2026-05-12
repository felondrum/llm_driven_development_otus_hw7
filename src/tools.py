import httpx
import os
from langchain_classic.tools import Tool


class DetectLanguageTool:
    """Tool for detecting language of input text."""
    
    name = "detect_language"
    description = "Определяет язык входного текста"
    
    def _run(self, text: str) -> str:
        """Detect language of the text."""
        # Mock implementation - in production use langdetect or similar
        lang_map = {
            "ru": "Russian",
            "en": "English", 
            "zh": "Chinese",
            "ar": "Arabic",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
        }
        # Simple heuristic based on characters
        if any(ord(c) > 127 for c in text):
            if any('а' <= c.lower() <= 'я' for c in text):
                return "ru"
            elif any('\u4e00' <= c <= '\u9fff' for c in text):
                return "zh"
            elif any('\u0600' <= c <= '\u06ff' for c in text):
                return "ar"
        return "en"
    
    def __call__(self, text: str) -> str:
        return self._run(text)


class TranslateTextTool:
    """Tool for translating text between languages."""
    
    name = "translate_text"
    description = "Переводит текст с одного языка на другой"
    
    def _run(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """Translate text from source to target language."""
        # Mock implementation - in production use LibreTranslate or similar API
        return f"[Translated] {text} -> {target_lang}"
    
    def __call__(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        return self._run(text, source_lang, target_lang)


class CurrencyConverterTool:
    """Tool for currency conversion using real-time rates."""
    
    name = "currency_converter"
    description = "Конвертирует валюту по текущему курсу"
    
    def _run(self, amount: float, from_currency: str, to_currency: str) -> str:
        """Convert currency amount using real API."""
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
        try:
            import asyncio
            async def fetch_rate():
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=5.0)
                    return response.json()
            
            data = asyncio.run(fetch_rate())
            rate = data["rates"].get(to_currency.upper())
            if rate:
                converted = amount * rate
                return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()} (rate: 1 {from_currency.upper()} = {rate} {to_currency.upper()})"
            return f"Currency {to_currency.upper()} not found"
        except Exception as e:
            return f"Error fetching rate: {str(e)}"
    
    def __call__(self, amount: float, from_currency: str, to_currency: str) -> str:
        return self._run(amount, from_currency, to_currency)


# LangChain Tool wrappers for agent integration
detect_language_tool = Tool(
    name="Detect Language",
    func=DetectLanguageTool(),
    description="Определяет язык входного текста. Используй, когда пользователь спрашивает о языке или нужно определить язык перед переводом."
)

translate_text_tool = Tool(
    name="Translate Text",
    func=lambda x, y="en": TranslateTextTool()(x, target_lang=y),
    description="Переводит текст на указанный язык. Формат: 'текст', 'язык'. Пример: 'Hello', 'ru'."
)

get_currency_rate_tool = Tool(
    name="Currency Converter",
    func=lambda x, y: CurrencyConverterTool()(float(x.split()[0]) if ' ' in x else float(x), x.split()[-2] if ' ' in x else "USD", x.split()[-1] if ' ' in x else "EUR"),
    description="Получает текущий курс валют. Принимает два кода валют (например, 'USD', 'RUB')."
)


def get_all_tools():
    """Return list of all available tools."""
    return [
        DetectLanguageTool(),
        TranslateTextTool(),
        CurrencyConverterTool(),
    ]