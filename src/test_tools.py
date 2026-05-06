"""
Test script for LangChain tools without requiring the full model.
"""

from tools import DetectLanguageTool, TranslateTextTool, CurrencyConverterTool


def test_detect_language():
    """Test language detection."""
    tool = DetectLanguageTool()
    
    test_cases = [
        ("Hello, how are you?", "en"),
        ("Привет, как дела?", "ru"),
        ("你好，你好吗？", "zh"),
        ("مرحبا، كيف حالك؟", "ar"),
    ]
    
    print("=== Language Detection Tests ===\n")
    for text, expected in test_cases:
        result = tool._run(text)
        status = "✓" if result == expected else "?"
        print(f"{status} '{text[:30]}...' -> {result} (expected: {expected})")
    print()


def test_translate():
    """Test translation."""
    tool = TranslateTextTool()
    
    print("=== Translation Tests ===\n")
    
    texts = [
        "Hello world",
        "Привет мир",
    ]
    
    for text in texts:
        result = tool._run(text, target_lang="en")
        print(f"  {result}")
    print()


def test_currency_converter():
    """Test currency conversion."""
    tool = CurrencyConverterTool()
    
    print("=== Currency Conversion Tests ===\n")
    
    conversions = [
        (100, "USD", "EUR"),
        (50, "EUR", "RUB"),
        (1000, "RUB", "USD"),
        (1, "GBP", "JPY"),
    ]
    
    for amount, from_curr, to_curr in conversions:
        result = tool._run(amount, from_curr, to_curr)
        print(f"  {result}")
    print()


if __name__ == "__main__":
    test_detect_language()
    test_translate()
    test_currency_converter()
    print("All tests completed!")
