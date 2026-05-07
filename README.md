# Многоязычный диалоговый ассистент с LoRA и LangChain Tools

## Цель работы
Улучшить качество диалогов большой языковой модели на разных языках с помощью 
параметрически-эффективного fine-tuning (LoRA) и расширить возможности модели 
через интеграцию с внешними API (перевод, определение языка, курсы валют, часовые пояса).

## Используемые технологии
- **Ollama** — локальный запуск LLM (qwen2.5:1.5b)
- **PEFT (LoRA)** — адаптация модели под диалоговый формат на 35 языках
- **LangChain** — создание инструментов (tools) для взаимодействия с внешним миром
- **Datasets (OpenAssistant oasst1)** — обучающий датасет с 84.4k диалогов
- **Hugging Face Transformers** — базовая модель Qwen2.5-1.5B-Instruct
- **FastAPI** — REST API для взаимодействия с моделью
- **Docker & docker-compose** — контейнеризация и оркестрация
- **uv** — быстрый менеджер пакетов Python

## Архитектура приложения

### Общая схема

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   FastAPI App   │────▶│  LangChain Agent │────▶│  External Tools │
│   (port 8000)   │     │  (Zero-shot ReAct)│     │  (Currency, etc)│
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
    ┌────────────┐     ┌────────────┐
    │  Ollama    │     │  Ollama    │
    │   Base     │     │   LoRA     │
    │ port 11434 │     │ port 11435 │
    │qwen2.5:1.5b│     │ qwen-lora  │
    └────────────┘     └────────────┘
         ▲                    ▲
         │                    │
         └────────────────────┘
              Параллельные
              запросы через
           /chat/compare API
```

### Dual-Model архитектура для сравнения

Проект использует **два контейнера Ollama** для одновременного сравнения:

| Контейнер | Порт | Модель | Назначение |
|-----------|------|--------|------------|
| `ollama-base` | 11434 | `qwen2.5:1.5b` | Базовая модель (оригинал) |
| `ollama-lora` | 11435 | `qwen-lora` | Дообученная модель (после LoRA) |

**Преимущества такой архитектуры:**
- 🚀 **Параллельное сравнение**: один запрос → два ответа одновременно
- 📊 **Визуализация разницы**: мгновенное сравнение качества ответов
- 🔬 **A/B тестирование**: одинаковые промпты для обеих моделей
- ⏱️ **Замер latency**: сравнение скорости генерации

---

## Пошаговая инструкция

### Шаг 1: Создание контейнеров Ollama

Запустите два контейнера для базовой и дообученной моделей:

```bash
docker compose up -d
```

Проверьте статус:

```bash
docker compose ps
```

Должны быть запущены:
- `ollama-base` (порт 11434)
- `ollama-lora` (порт 11435)

### Шаг 2: Загрузка базовой модели

Загрузите модель Qwen2.5-1.5B в контейнер `ollama-base`:

```bash
docker exec ollama-base ollama pull qwen2.5:1.5b
```

⏱️ **Время загрузки:** ~5-10 минут в зависимости от скорости интернета

Проверьте успешность:

```bash
docker exec ollama-base ollama list
```

Должна отображаться модель `qwen2.5:1.5b`.

### Шаг 3: Обучение на датасете (LoRA Fine-tuning)

Запустите обучение адаптера LoRA на датасете OpenAssistant oasst1:

```bash
# Быстрое тестирование (10 примеров, 1 эпоха)
python src/train.py --max-samples 10 --epochs 1

# Полное обучение (рекомендуется для Mac M1 Pro)
python src/train.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --batch-size 4 --max-samples 1000
```

**Параметры обучения:**

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | Базовая модель |
| `--max-samples` | `10` | Количество примеров для обучения |
| `--epochs` | `1` | Количество эпох обучения |
| `--batch-size` | `2` | Размер батча |
| `--output` | `./lora_adapter` | Папка для сохранения адаптера |

**Результат обучения:**
- Адаптер LoRA сохраняется в папку `./lora_adapter`
- Веса базовой модели не изменяются

⏱️ **Время обучения:**
- 10 примеров: ~2-5 минут
- 1000 примеров: ~30-60 минут (на Mac M1 Pro)

### Шаг 4: Добавление адаптера в Ollama

После завершения обучения запустите скрипт слияния и импорта:

```bash
python src/merge_and_export.py
```

**Что делает скрипт:**
1. ✅ Загружает базовую модель `Qwen/Qwen2.5-1.5B-Instruct`
2. ✅ Применяет адаптеры LoRA из `./lora_adapter`
3. ✅ Сливает веса (Base + LoRA) → `./merged_model`
4. ✅ Копирует модель в контейнер `ollama-lora`
5. ✅ Создаёт модель `qwen-lora` в Ollama

⏱️ **Время выполнения:** ~5-10 минут

**Проверка успешности:**

```bash
docker exec ollama-lora ollama list
```

Должна отображаться модель `qwen-lora`.

### Шаг 5: Тестовые прогоны

#### Простой запрос к базовой модели:

```bash
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:1.5b",
  "prompt": "Расскажи короткую историю про космос на русском языке.",
  "stream": false
}'
```

#### Простой запрос к LoRA модели:

```bash
curl -X POST http://localhost:11435/api/generate -d '{
  "model": "qwen-lora",
  "prompt": "Расскажи короткую историю про космос на русском языке.",
  "stream": false
}'
```

#### Сравнение моделей через API:

Запустите FastAPI сервер для доступа к endpoint `/chat/compare`:

```bash
# Установка зависимостей (если ещё не установлены)
uv sync
uv pip install trl accelerate

# Запуск API сервера
uv run python -m src.api
```

Теперь используйте endpoint сравнения:

```bash
curl -X POST http://localhost:8000/chat/compare -H "Content-Type: application/json" -d '{
  "message": "Переведи фразу '\''Hello world'\'' на французский язык."
}' | jq .
```

**Ответ будет содержать оба ответа одновременно:**

```json
{
  "base_response": "...",
  "lora_response": "...",
  "base_model": "qwen2.5:1.5b",
  "lora_model": "qwen-lora",
  "base_latency_ms": 1234,
  "lora_latency_ms": 1156
}
```

---

## Запуск API и использование

После того как модели обучены и импортированы, запустите FastAPI сервер для удобной работы:

### Запуск API сервера

```bash
# Активация окружения и запуск
uv sync
uv run python -m src.api
```

Сервер запустится на порту `8000`.

### Доступные endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/health` | Проверка здоровья сервиса |
| GET | `/models/status` | Статус обеих моделей (base и lora) |
| POST | `/chat` | Запрос к базовой модели |
| POST | `/chat/lora` | Запрос к LoRA модели |
| POST | `/chat/compare` | **Параллельный запрос к обеим моделям** |

### Примеры использования API

#### 1. Проверка статуса моделей

```bash
curl http://localhost:8000/models/status | jq .
```

#### 2. Запрос к базовой модели

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "Расскажи короткую историю про космос на русском языке."
}'
```

#### 3. Запрос к LoRA модели

```bash
curl -X POST http://localhost:8000/chat/lora -H "Content-Type: application/json" -d '{
  "message": "Расскажи историю про кота на немецком языке."
}'
```

#### 4. Сравнение моделей (параллельный запрос)

```bash
curl -X POST http://localhost:8000/chat/compare -H "Content-Type: application/json" -d '{
  "message": "Какой сейчас курс доллара (USD) к рублю (RUB)?"
}' | jq .
```

#### 5. Использование инструментов (Tools)

Автоматическое распознавание намерений и вызов инструментов:

```bash
# Курс валют
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "Какой сейчас курс доллара (USD) к рублю (RUB)?"
}'

# Перевод текста
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "Переведи фразу '\''Hello world'\'' на русский язык."
}'

# Определение языка
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "На каком языке написано: '\''Guten Tag'\''?"
}'
```

### Формат запроса

```json
{
  "message": "Текст сообщения",
  "history": [
    {"role": "user", "content": "Предыдущий вопрос"},
    {"role": "assistant", "content": "Предыдущий ответ"}
  ]
}
```

### Формат ответа

```json
{
  "response": "Ответ модели",
  "source": "direct или agent",
  "model": "qwen2.5:1.5b или qwen-lora",
  "latency_ms": 1234.56
}
```

---

## Структура проекта

```
/workspace
├── docker-compose.yml      # Конфигурация Docker Compose (2x Ollama)
├── pyproject.toml          # Зависимости Python
├── README.md               # Документация
└── src/
    ├── api.py              # FastAPI приложение (dual-model API)
    ├── tools.py            # LangChain инструменты (3 tools)
    ├── train.py            # Скрипт обучения LoRA (Mac M1 оптимизирован)
    ├── merge_and_export.py # Слияние LoRA + импорт в Ollama
    ├── inference.py        # Инференс с LoRA адаптером
    └── test_tools.py       # Тесты инструментов
```

---

## Troubleshooting

### Модель не загружается в Ollama

```bash
# Проверьте доступность Ollama
docker exec ollama-base ollama list

# Переустановите модель
docker exec ollama-base ollama rm qwen2.5:1.5b
docker exec ollama-base ollama pull qwen2.5:1.5b
```

### Ошибки памяти при обучении LoRA

- Уменьшите `--batch-size` до 1 или 2
- Уменьшите `--max-samples`
- Используйте параметры по умолчанию для Mac M1 Pro

### Проблемы с контейнерами

```bash
# Перезапуск контейнеров
docker compose restart

# Просмотр логов
docker compose logs ollama-base
docker compose logs ollama-lora

# Полный сброс
docker compose down
docker compose up -d
```

### Проверка доступности моделей

```bash
# Базовая модель
docker exec ollama-base ollama list

# LoRA модель (после импорта)
docker exec ollama-lora ollama list
```

---

## Лицензия

MIT License
