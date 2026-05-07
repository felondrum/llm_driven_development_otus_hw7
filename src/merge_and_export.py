"""
Скрипт для слияния базовой модели и адаптеров LoRA,
и экспорта результата для использования в Ollama.

Этот скрипт выполняет критически важный шаг: объединяет веса базовой модели
с обученными адаптерами LoRA, создавая полноценную модель, которую можно
использовать в Ollama вместо базовой версии.

После завершения train.py запустите этот скрипт для импорта адаптера в контейнер.
"""
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import shutil
import subprocess

# Конфигурация
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_ADAPTER_PATH = "./lora_adapter"  # Совпадает с output_dir по умолчанию в train.py
OUTPUT_DIR = "./merged_model"
OLLAMA_LORA_CONTAINER = "ollama-lora"
LORA_MODEL_NAME = "qwen-lora"

def merge_and_export():
    print("🚀 Запуск слияния модели и адаптеров LoRA...")

    # Проверка существования адаптера
    if not os.path.exists(LORA_ADAPTER_PATH):
        raise FileNotFoundError(
            f"Папка с адаптерами не найдена: {LORA_ADAPTER_PATH}. "
            "Сначала запустите обучение: python src/train.py"
        )

    # 1. Загрузка базовой модели и токенизатора
    print(f"📥 Загрузка базовой модели: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Применение адаптеров LoRA
    print(f"🔗 Применение адаптеров из: {LORA_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    
    # 3. Слияние весов (Merge)
    print("🔨 Слияние весов (LoRA + Base)...")
    merged_model = model.merge_and_unload()
    
    # 4. Сохранение слитой модели в формате Transformers
    print(f"💾 Сохранение слитой модели в: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 5. Подготовка структуры для Ollama Modelfile
    print("📦 Подготовка к экспорту для Ollama...")
    os.makedirs("./ollama_model", exist_ok=True)
    
    # Создаем Modelfile для Ollama
    modelfile_content = f"""FROM {OUTPUT_DIR}
PARAMETER temperature 0.7
SYSTEM "You are a helpful multilingual assistant trained with LoRA on OpenAssistant dataset."
"""
    modelfile_path = "./ollama_model/Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
        
    print("\n" + "="*60)
    print("✅ Слияние завершено! Модель готова к импорту в Ollama.")
    print("="*60)
    print(f"📁 Путь к слитой модели: {OUTPUT_DIR}")
    print(f"📄 Modelfile создан: {modelfile_path}")
    
    # 6. Автоматический импорт в контейнер ollama-lora
    print("\n🔄 Импорт в контейнер ollama-lora...")
    
    try:
        # Копирование модели в контейнер
        print(f"📤 Копирование модели в контейнер {OLLAMA_LORA_CONTAINER}...")
        subprocess.run([
            "docker", "cp", OUTPUT_DIR, 
            f"{OLLAMA_LORA_CONTAINER}:/tmp/merged_model"
        ], check=True)
        
        # Копирование Modelfile в контейнер
        subprocess.run([
            "docker", "cp", modelfile_path,
            f"{OLLAMA_LORA_CONTAINER}:/tmp/Modelfile"
        ], check=True)
        
        # Создание модели в Ollama
        print(f"🏗️  Создание модели '{LORA_MODEL_NAME}' в Ollama...")
        subprocess.run([
            "docker", "exec", OLLAMA_LORA_CONTAINER,
            "ollama", "create", LORA_MODEL_NAME, "-f", "/tmp/Modelfile"
        ], check=True)
        
        print("\n" + "="*60)
        print(f"✅ Модель '{LORA_MODEL_NAME}' успешно импортирована в ollama-lora!")
        print("="*60)
        print(f"🌐 Доступна через: http://localhost:11435")
        print(f"📊 API для сравнения: POST http://localhost:8000/chat/compare")
        print("="*60)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при импорте в Ollama: {e}")
        print("\n📋 Попробуйте вручную:")
        print(f"   docker cp {OUTPUT_DIR} {OLLAMA_LORA_CONTAINER}:/tmp/merged_model")
        print(f"   docker cp {modelfile_path} {OLLAMA_LORA_CONTAINER}:/tmp/Modelfile")
        print(f"   docker exec {OLLAMA_LORA_CONTAINER} ollama create {LORA_MODEL_NAME} -f /tmp/Modelfile")
        raise

if __name__ == "__main__":
    merge_and_export()
