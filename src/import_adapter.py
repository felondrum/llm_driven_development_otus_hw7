import requests
import os
import time
from pathlib import Path

OLLAMA_LORA_URL = "http://localhost:11435"
ADAPTER_DIR_NAME = "lora_adapter"  # Имя папки, совпадающее с output_dir в train.py
NEW_MODEL_NAME = "qwen2.5-lora-custom"

def wait_for_ollama():
    print(f"Ожидание готовности Ollama на {OLLAMA_LORA_URL}...")
    while True:
        try:
            r = requests.get(f"{OLLAMA_LORA_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                print("Ollama LoRA контейнер готов.")
                break
        except Exception as e:
            pass
        time.sleep(2)

def import_model():
    wait_for_ollama()
    
    # Путь внутри контейнера (см. docker-compose.yml volume)
    container_adapter_path = f"/root/adapters/{ADAPTER_DIR_NAME}"
    
    # Проверяем существование адаптера локально
    local_adapter_path = Path("./adapters") / ADAPTER_DIR_NAME
    if not local_adapter_path.exists():
        print(f"Ошибка: Адаптер не найден в {local_adapter_path}")
        print("Сначала запустите обучение (train.py)")
        return
    
    # Создаем Modelfile динамически
    modelfile_content = f"""FROM qwen2.5:3b
ADAPTER {container_adapter_path}
PARAMETER temperature 0.7
SYSTEM "Вы - полезный ассистент, дообученный на пользовательском датасете."
"""
    
    print(f"Создание модели {NEW_MODEL_NAME} с адаптером из {container_adapter_path}...")
    
    payload = {
        "name": NEW_MODEL_NAME,
        "modelfile": modelfile_content,
        "stream": False
    }
    
    try:
        resp = requests.post(f"{OLLAMA_LORA_URL}/api/create", json=payload, timeout=120)
        if resp.status_code == 200:
            print(f"✅ Успешно! Модель '{NEW_MODEL_NAME}' создана и готова к использованию.")
            print(f"   Запрос: curl http://localhost:11435/api/generate -d '{{\"model\": \"{NEW_MODEL_NAME}\", \"prompt\": \"Привет\"}}'")
        else:
            print(f"❌ Ошибка создания: {resp.text}")
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")

if __name__ == "__main__":
    import_model()
