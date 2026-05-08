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
    # Важно: НЕ используем device_map="auto", чтобы избежать meta-тензоров
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None,  # Критично: загружаем без device_map
        trust_remote_code=True
    )
    
    # Перемещаем модель на устройство после загрузки
    if torch.backends.mps.is_available():
        base_model = base_model.to("mps")
    elif torch.cuda.is_available():
        base_model = base_model.to("cuda")
    else:
        base_model = base_model.to("cpu")

    # 2. Применение адаптеров LoRA
    print(f"🔗 Применение адаптеров из: {LORA_ADAPTER_PATH}")
    # is_trainable=False важно для корректного слияния
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH, is_trainable=False)
    
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
    
    gguf_output = os.path.join(OUTPUT_DIR, "model.gguf")
    
    # Пробуем найти скрипт конвертации из llama.cpp
    convert_script = None
    possible_paths = [
        "./convert-hf-to-gguf.py",
        "../llama.cpp/convert-hf-to-gguf.py",
        os.path.expanduser("~/llama.cpp/convert-hf-to-gguf.py"),
        "./llama.cpp/convert-hf-to-gguf.py",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            convert_script = path
            break
    
    # Если скрипт не найден, пробуем найти через установленный пакет llama-cpp-python
    if convert_script is None:
        try:
            import llama_cpp
            # Пытаемся найти скрипт в директории пакета
            package_dir = os.path.dirname(llama_cpp.__file__)
            # Пробуем разные варианты именования скрипта
            potential_scripts = [
                os.path.join(package_dir, "..", "bin", "convert-hf-to-gguf.py"),
                os.path.join(package_dir, "..", "bin", "convert_hf_to_gguf.py"),
                os.path.join(package_dir, "convert-hf-to-gguf.py"),
                os.path.join(package_dir, "convert_hf_to_gguf.py"),
            ]
            for potential_script in potential_scripts:
                if os.path.exists(potential_script):
                    convert_script = os.path.abspath(potential_script)
                    print(f"📄 Найден скрипт конвертации в пакете: {convert_script}")
                    break
        except ImportError:
            pass
    
    # Если всё ещё не найден, пробуем проверить установку через pip show
    if convert_script is None:
        try:
            result = subprocess.run(
                ["pip", "show", "llama-cpp-python"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                # Ищем путь к установке в выводе pip show
                for line in result.stdout.split('\n'):
                    if line.startswith('Location:'):
                        location = line.split(':', 1)[1].strip()
                        potential_script = os.path.join(location, "llama_cpp", "convert-hf-to-gguf.py")
                        if os.path.exists(potential_script):
                            convert_script = potential_script
                            print(f"📄 Найден скрипт конвертации: {convert_script}")
                        break
        except Exception:
            pass
    
    # Если скрипт найден, используем его
    if convert_script:
        print(f"🚀 Запуск конвертации через: {convert_script}")
        
        # Проверяем наличие sentencepiece для Qwen моделей
        try:
            import sentencepiece
        except ImportError:
            print("⚠️  Для конвертации моделей Qwen требуется пакет sentencepiece")
            print("   Установите: pip install sentencepiece")
            raise ImportError("Требуется пакет sentencepiece. Установите: pip install sentencepiece")
        
        result = subprocess.run([
            "python3", convert_script,
            OUTPUT_DIR,
            "--outfile", gguf_output,
            "--outtype", "f16"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Ошибка конвертации:\n{result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, result.args)
    else:
        # Скрипт не найден — предоставляем подробные инструкции
        print("⚠️  Скрипт convert-hf-to-gguf.py не найден.")
        print("\n📋 Для конвертации модели в GGUF формат необходимо установить llama.cpp:")
        print("\n--- Вариант 1: Сборка через CMake (рекомендуется) ---")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp")
        print("   cmake -B build")
        print("   cmake --build build --config Release")
        print("\n--- Вариант 2: Сборка через make (альтернатива) ---")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp && make")
        print("\nПосле сборки запустите конвертацию вручную:")
        print(f"   python llama.cpp/convert-hf-to-gguf.py {OUTPUT_DIR} --outfile {gguf_output}")
        print("\nИли укажите путь к существующей установке llama.cpp:")
        print("   export LLAMA_CPP_PATH=/path/to/llama.cpp")
        raise FileNotFoundError(
            "Скрипт convert-hf-to-gguf.py не найден. Установите llama.cpp."
        )
    
    # Проверка созданного файла
    if not os.path.exists(gguf_output):
        raise FileNotFoundError(f"GGUF файл не был создан: {gguf_output}")
    
    file_size_gb = os.path.getsize(gguf_output) / (1024 ** 3)
    print(f"✅ GGUF файл создан: {gguf_output} ({file_size_gb:.2f} GB)")

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
