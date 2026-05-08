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
import json

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
    
    # Используем float32 для стабильности при слиянии
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
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
    
    # 5. Очистка config.json от PEFT-метаданных
    print("🧹 Очистка config.json от метаданных PEFT...")
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Удаляем поля, связанные с PEFT/LoRA
    fields_to_remove = [
        "adapter_config", "peft_type", "base_model_name_or_path", 
        "task_type", "inference_mode"
    ]
    for field in fields_to_remove:
        config.pop(field, None)
    
    # Сохраняем чистый конфиг
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Config.json очищен: {config_path}")

    # 6. Конвертация в GGUF формат
    print("🔄 Конвертация в GGUF формат...")
    
    # Проверяем наличие скрипта конвертации
    convert_script = None
    possible_paths = [
        "./convert-hf-to-gguf.py",
        "../llama.cpp/convert-hf-to-gguf.py",
        "/usr/local/lib/python3.*/site-packages/llama_cpp/convert-hf-to-gguf.py"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            convert_script = path
            break
    
    # Если не найден локально, пробуем найти через llama.cpp в системе
    if convert_script is None:
        try:
            result = subprocess.run(
                ["python3", "-c", "import llama_cpp; print(llama_cpp.__file__)"],
                capture_output=True, text=True, check=True
            )
            llama_cpp_dir = os.path.dirname(result.stdout.strip())
            potential_script = os.path.join(llama_cpp_dir, "..", "bin", "convert-hf-to-gguf.py")
            if os.path.exists(potential_script):
                convert_script = potential_script
        except:
            pass
    
    # Если всё ещё не найден, используем встроенный путь или скачиваем
    if convert_script is None:
        print("⚠️  Скрипт convert-hf-to-gguf.py не найден. Попытка использовать через llama-cpp-python...")
        # Пробуем вызвать конвертацию через модуль
        gguf_output = os.path.join(OUTPUT_DIR, "model.gguf")
        
        # Альтернатива: используем subprocess с python -m
        try:
            subprocess.run([
                "python3", "-m", "llama_cpp.convert", 
                OUTPUT_DIR, 
                "--outfile", gguf_output,
                "--outtype", "f16"
            ], check=True)
            print(f"✅ GGUF файл создан: {gguf_output}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка конвертации через llama_cpp.convert: {e}")
            print("📋 Установите llama-cpp-python или скачайте llama.cpp:")
            print("   pip install llama-cpp-python")
            print("   или")
            print("   git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
            raise
    else:
        gguf_output = os.path.join(OUTPUT_DIR, "model.gguf")
        subprocess.run([
            "python3", convert_script,
            OUTPUT_DIR,
            "--outfile", gguf_output,
            "--outtype", "f16"
        ], check=True)
        print(f"✅ GGUF файл создан: {gguf_output}")

    # 7. Подготовка структуры для Ollama Modelfile
    print("📦 Подготовка к экспорту для Ollama...")
    os.makedirs("./ollama_model", exist_ok=True)
    
    # Создаем Modelfile для Ollama с путем к GGUF файлу
    modelfile_content = f"""FROM /tmp/merged_model/model.gguf
PARAMETER temperature 0.7
SYSTEM \"You are a helpful multilingual assistant trained with LoRA on OpenAssistant dataset.\"
"""
    modelfile_path = "./ollama_model/Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
        
    print("\n" + "="*60)
    print("✅ Слияние завершено! Модель готова к импорту в Ollama.")
    print("="*60)
    print(f"📁 Путь к слитой модели: {OUTPUT_DIR}")
    print(f"📄 GGUF файл: {gguf_output}")
    print(f"📄 Modelfile создан: {modelfile_path}")
    
    # 8. Автоматический импорт в контейнер ollama-lora
    print("\n🔄 Импорт в контейнер ollama-lora...")
    
    try:
        # Копирование модели в контейнер (копируем содержимое, а не папку)
        print(f"📤 Копирование модели в контейнер {OLLAMA_LORA_CONTAINER}...")
        
        # Сначала удаляем старую директорию в контейнере если существует
        subprocess.run([
            "docker", "exec", OLLAMA_LORA_CONTAINER,
            "rm", "-rf", "/tmp/merged_model"
        ], check=False)
        
        # Копируем всю директорию
        subprocess.run([
            "docker", "cp", OUTPUT_DIR, 
            f"{OLLAMA_LORA_CONTAINER}:/tmp/merged_model"
        ], check=True)
        
        # Копирование Modelfile в контейнер
        subprocess.run([
            "docker", "cp", modelfile_path,
            f"{OLLAMA_LORA_CONTAINER}:/tmp/Modelfile"
        ], check=True)
        
        # Проверяем права доступа к файлам в контейнере
        print("🔒 Проверка прав доступа к файлам...")
        subprocess.run([
            "docker", "exec", OLLAMA_LORA_CONTAINER,
            "chmod", "-R", "755", "/tmp/merged_model"
        ], check=False)
        
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
