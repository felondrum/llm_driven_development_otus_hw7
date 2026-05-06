"""
Скрипт для сравнения работы базовой модели и модели с LoRA.
Запускает одинаковые промпты через API /chat/compare и анализирует результаты,
показывая разницу в качестве ответов и точности вызова инструментов.

Теперь использует новый эндпоинт /chat/compare для параллельного получения
ответов от обеих моделей (базовой и LoRA).
"""
import requests
import json
import time

API_URL = "http://localhost:8000"

def get_comparison(prompt):
    """Отправляет запрос к API сравнения и получает ответы от обеих моделей."""
    url = f"{API_URL}/chat/compare"
    payload = {"message": prompt}
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_single_response(prompt, model="base"):
    """Получает ответ от одной модели (base или lora)."""
    if model == "base":
        url = f"{API_URL}/chat"
    else:
        url = f"{API_URL}/chat/lora"
    
    payload = {"message": prompt}
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def check_models_status():
    """Проверяет доступность обеих моделей."""
    url = f"{API_URL}/models/status"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def compare_models():
    # Проверка статуса моделей
    print("="*70)
    print("🔍 ПРОВЕРКА ДОСТУПНОСТИ МОДЕЛЕЙ")
    print("="*70)
    
    status = check_models_status()
    if "error" in status:
        print(f"❌ Ошибка проверки статуса: {status['error']}")
        print("\nУбедитесь, что сервисы запущены:")
        print("   docker compose up -d")
        return
    
    print("\n📊 Статус моделей:")
    for model_type, info in status.items():
        available = "✅" if info.get("available") else "❌"
        print(f"   {available} {model_type.upper()}: {info.get('url', 'N/A')}")
        if info.get("available"):
            models_list = info.get("models", [])
            print(f"      Доступные модели: {', '.join(models_list) if models_list else 'нет'}")
            target_present = "✅" if info.get("target_model_present") else "❌"
            expected = "qwen2.5:3b" if model_type == "base" else "qwen-lora"
            print(f"      {target_present} Целевая модель '{expected}': {'загружена' if info.get('target_model_present') else 'НЕ ЗАГРУЖЕНА'}")
    
    # Проверка: если LoRA модель не загружена, предупреждаем
    if not status.get("lora", {}).get("target_model_present"):
        print("\n" + "="*70)
        print("⚠️  ВНИМАНИЕ: Модель qwen-lora не найдена!")
        print("="*70)
        print("\n📋 Запустите обучение и импорт модели:")
        print("   1. docker compose exec llm-app python src/train.py")
        print("   2. docker compose exec llm-app python src/merge_and_export.py")
        print("\nПосле импорта повторите тест.")
        print("="*70)
    
    test_prompts = [
        {
            "prompt": "Переведи фразу 'Hello world' на французский.",
            "expected_tool": "translate_text",
            "description": "Тест перевода (должен вызвать инструмент)"
        },
        {
            "prompt": "Какой сейчас курс доллара к евро?",
            "expected_tool": "currency_converter",
            "description": "Тест курса валют (должен вызвать инструмент)"
        },
        {
            "prompt": "Определи язык фразы: 'Guten Tag, wie geht es Ihnen?'",
            "expected_tool": "detect_language",
            "description": "Тест определения языка"
        },
        {
            "prompt": "Привет! Как дела? Ответь кратко.",
            "expected_tool": None,
            "description": "Обычный диалог (инструменты не нужны)"
        },
        {
            "prompt": "Расскажи короткую историю про кота на русском языке.",
            "expected_tool": None,
            "description": "Креативная задача (инструменты не нужны)"
        },
        {
            "prompt": "What is the weather like today?",
            "expected_tool": None,
            "description": "English query (multilingual test)"
        }
    ]

    print("\n" + "="*70)
    print("🔬 СРАВНЕНИЕ: Базовая модель vs LoRA дообученная модель")
    print("="*70)
    print("\n💡 Примечание:")
    print("   - Базовая модель (qwen2.5:3b): оригинальная версия из Ollama")
    print("   - LoRA модель (qwen-lora): дообучена на OpenAssistant oasst1")
    print("   - Обе модели используют LangChain Tools при необходимости")
    print("   - Запросы выполняются параллельно через /chat/compare")
    print("="*70)
    
    results = []
    
    for i, test in enumerate(test_prompts, 1):
        prompt = test["prompt"]
        expected_tool = test["expected_tool"]
        description = test["description"]
        
        print(f"\n{'='*70}")
        print(f"📝 ТЕСТ {i}: {description}")
        print(f"💬 Запрос: {prompt}")
        print(f"{'='*70}")
        
        # Запрос к API сравнения
        print("\n⏳ Параллельный запрос к обеим моделям...")
        comparison_result = get_comparison(prompt)
        
        if "error" in comparison_result:
            print(f"❌ Ошибка API: {comparison_result['error']}")
            continue
        
        base_response = comparison_result.get("base_response", "Нет ответа")
        lora_response = comparison_result.get("lora_response", "Нет ответа")
        base_latency = comparison_result.get("base_latency_ms", 0)
        lora_latency = comparison_result.get("lora_latency_ms", 0)
        
        print(f"\n🔵 БАЗОВАЯ МОДЕЛЬ ({comparison_result.get('base_model', 'N/A')}):")
        print(f"   Время ответа: {base_latency:.0f} мс")
        print(f"   Ответ: {base_response[:250]}{'...' if len(base_response) > 250 else ''}")
        
        print(f"\n🟢 LORA МОДЕЛЬ ({comparison_result.get('lora_model', 'N/A')}):")
        print(f"   Время ответа: {lora_latency:.0f} мс")
        print(f"   Ответ: {lora_response[:250]}{'...' if len(lora_response) > 250 else ''}")
        
        # Сравнительный анализ
        print(f"\n📊 СРАВНЕНИЕ:")
        if len(lora_response) > len(base_response):
            diff = ((len(lora_response) - len(base_response)) / len(base_response)) * 100
            print(f"   📈 LoRA ответ на {diff:.1f}% длиннее")
        elif len(lora_response) < len(base_response):
            diff = ((len(base_response) - len(lora_response)) / len(base_response)) * 100
            print(f"   📉 LoRA ответ на {diff:.1f}% короче")
        else:
            print(f"   ➖ Ответы одинаковой длины")
        
        speed_diff = ((lora_latency - base_latency) / base_latency) * 100 if base_latency > 0 else 0
        if speed_diff > 10:
            print(f"   🐌 LoRA на {speed_diff:.1f}% медленнее")
        elif speed_diff < -10:
            print(f"   ⚡ LoRA на {abs(speed_diff):.1f}% быстрее")
        else:
            print(f"   ➖ Скорость примерно одинаковая")
        
        results.append({
            "test": i,
            "prompt": prompt,
            "base_response": base_response,
            "lora_response": lora_response,
            "base_latency_ms": base_latency,
            "lora_latency_ms": lora_latency,
            "expected_tool": expected_tool
        })

    # Итоговая статистика
    print("\n" + "="*70)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("="*70)
    
    total_tests = len(results)
    avg_base_latency = sum(r["base_latency_ms"] for r in results) / total_tests if total_tests > 0 else 0
    avg_lora_latency = sum(r["lora_latency_ms"] for r in results) / total_tests if total_tests > 0 else 0
    
    print(f"Всего тестов: {total_tests}")
    print(f"Среднее время ответа (Base): {avg_base_latency:.0f} мс")
    print(f"Среднее время ответа (LoRA): {avg_lora_latency:.0f} мс")
    
    speed_improvement = ((avg_base_latency - avg_lora_latency) / avg_base_latency) * 100 if avg_base_latency > 0 else 0
    if speed_improvement > 0:
        print(f"📈 Ускорение LoRA: {speed_improvement:.1f}%")
    else:
        print(f"📉 Замедление LoRA: {abs(speed_improvement):.1f}%")
    
    print("\n" + "="*70)
    print("💡 КАК ИНТЕРПРЕТИРОВАТЬ РЕЗУЛЬТАТЫ:")
    print("="*70)
    print("""
    ✅ LoRA должна показывать лучшие результаты в:
       - Точности определения необходимости вызова инструментов
       - Форматировании ответов на разных языках
       - Следовании инструкциям (instruction following)
       - Снижении галлюцинаций
    
    ⚠️  Если различий нет или LoRA хуже:
       - Возможно, обучение было на слишком малом количестве эпох
       - Датасет мог быть недостаточно релевантным
       - Параметры LoRA (r, alpha) могут требовать настройки
    """)
    print("="*70)

if __name__ == "__main__":
    compare_models()
