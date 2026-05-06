"""
Скрипт для подготовки датасета OpenAssistant oasst1.
Используется для создания локального файла данных, если потребуется тонкая настройка.
"""
from datasets import load_dataset

def prepare_dataset():
    print("Loading OpenAssistant oasst1 dataset...")
    dataset = load_dataset("OpenAssistant/oasst1")

    # Фильтрация и подготовка
    train_data = dataset['train']

    print(f"Total samples: {len(train_data)}")

    # Пример сохранения в JSONL для будущего fine-tuning
    output_file = "data/oasst_prepared.jsonl"
    import os
    os.makedirs("data", exist_ok=True)

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in train_data.select(range(1000)): # Ограничим для примера
            if item['role'] == 'assistant':
                f.write(f"{item['text']}\n")
                count += 1

    print(f"Saved {count} assistant responses to {output_file}")

if __name__ == "__main__":
    prepare_dataset()