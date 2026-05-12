"""
Скрипт для экспорта модели в Docker контейнер с Ollama.

Этот скрипт копирует уже сконвертированную GGUF модель и Modelfile
в Docker контейнер ollama-lora, создаёт модель в Ollama и делает её
доступной через API.

Предварительное требование: должен быть выполнен merge.py
    python src/merge.py

Запуск:
    python src/export.py
"""
import os
import subprocess

# Конфигурация
OUTPUT_DIR = "./merged_model"
OLLAMA_LORA_CONTAINER = "ollama-lora"
LORA_MODEL_NAME = "qwen-lora"


def export_to_docker():
    print("🚀 Запуск экспорта модели в Docker/Ollama...")

    # Проверка существования GGUF файла
    gguf_output = os.path.join(OUTPUT_DIR, "model.gguf")
    if not os.path.exists(gguf_output):
        raise FileNotFoundError(
            f"GGUF файл не найден: {gguf_output}. "
            "Сначала запустите слияние модели: python src/merge.py"
        )

    # 1. Подготовка структуры для Ollama Modelfile
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
        
    print(f"📄 Modelfile создан: {modelfile_path}")

    # 2. Автоматический импорт в контейнер ollama-lora
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
    export_to_docker()
