import httpx
import os
from langchain.tools import Tool

# --- Detect Language Tool ---
async def detect_language_async(text: str) -> str:
    # Используем бесплатный API или заглушку для примера
    # В реальном проекте лучше использовать локальную модель или платный API
    return f"Detected language for '{text[:20]}...': ru (Russian) [Mock]"

def detect_language_wrapper(text: str) -> str:
    import asyncio
    return asyncio.run(detect_language_async(text))

detect_language_tool = Tool(
    name="Detect Language",
    func=detect_language_wrapper,
    description="Определяет язык входного текста. Используй, когда пользователь спрашивает о языке или нужно определить язык перед переводом."
)

# --- Translate Text Tool ---
async def translate_text_async(text: str, target_lang: str = "en") -> str:
    # Mock implementation. Replace with real API (e.g., LibreTranslate, Google Translate)
    return f"Translated '{text}' to {target_lang}: [Translation Mock]"

def translate_text_wrapper(text: str, target_lang: str = "en") -> str:
    import asyncio
    # Парсинг аргументов если они приходят одной строкой, но LangChain обычно передает их правильно
    return asyncio.run(translate_text_async(text, target_lang))

translate_text_tool = Tool(
    name="Translate Text",
    func=translate_text_wrapper,
    description="Переводит текст на указанный язык. Формат: 'текст', 'язык'. Пример: 'Hello', 'ru'."
)

# --- Currency Converter Tool ---
async def get_currency_rate_async(from_curr: str, to_curr: str) -> str:
    url = f"https://api.exchangerate-api.com/v4/latest/{from_curr.upper()}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0)
            data = response.json()
            rate = data["rates"].get(to_curr.upper())
            if rate:
                return f"Rate: 1 {from_curr.upper()} = {rate} {to_curr.upper()}"
            return "Currency not found."
    except Exception as e:
        return f"Error fetching rate: {str(e)}"

def get_currency_rate_wrapper(from_curr: str, to_curr: str) -> str:
    import asyncio
    return asyncio.run(get_currency_rate_async(from_curr, to_curr))

get_currency_rate_tool = Tool(
    name="Currency Converter",
    func=get_currency_rate_wrapper,
    description="Получает текущий курс валют. Принимает два кода валют (например, 'USD', 'RUB')."
)