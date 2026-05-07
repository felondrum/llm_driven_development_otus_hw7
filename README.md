# Многоязычный диалоговый ассистент с LoRA и LangChain Tools

## Цель работы
Улучшить качество диалогов большой языковой модели на разных языках с помощью 
параметрически-эффективного fine-tuning (LoRA) и расширить возможности модели 
через интеграцию с внешними API (перевод, определение языка, курсы валют, часовые пояса).

## Используемые технологии
- **Ollama** — локальный запуск LLM (qwen2.5:3b)
- **PEFT (LoRA)** — адаптация модели под диалоговый формат на 35 языках
- **LangChain** — создание инструментов (tools) для взаимодействия с внешним миром
- **Datasets (OpenAssistant oasst1)** — обучающий датасет с 84.4k диалогов
- **Hugging Face Transformers** — базовая модель Qwen2.5
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
    │ qwen2.5:3b │     │ qwen-lora  │
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
| `ollama-base` | 11434 | `qwen2.5:3b` | Базовая модель (оригинал) |
| `ollama-lora` | 11435 | `qwen-lora` | Дообученная модель (после LoRA) |

**Преимущества такой архитектуры:**
- 🚀 **Параллельное сравнение**: один запрос → два ответа одновременно
- 📊 **Визуализация разницы**: мгновенное сравнение качества ответов
- 🔬 **A/B тестирование**: одинаковые промпты для обеих моделей
- ⏱️ **Замер latency**: сравнение скорости генерации

## Этапы выполнения

### 1. Fine-tuning (LoRA)
Загрузка датасета OpenAssistant oasst1, токенизация, обучение LoRA адаптера, сравнение с базовой моделью.

**Скрипт обучения:** `src/train.py`

```bash
# Запуск обучения с параметрами по умолчанию
python -m src.train --model Qwen/Qwen2.5-3B-Instruct --epochs 3 --batch-size 4

# Обучение на ограниченном наборе данных (для тестирования)
python -m src.train --max-samples 1000 --epochs 1

# Обучение без 4-bit квантования (требует больше GPU памяти)
python -m src.train --no-4bit
```

**Параметры LoRA:**
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 2. Создание Tools
Реализация инструментов для расширения возможностей модели:

| Инструмент | Описание | API |
|------------|----------|-----|
| `detect_language` | Определение языка текста | Эвристический анализ символов |
| `translate_text` | Перевод текста | Mock (готов к интеграции с LibreTranslate) |
| `currency_converter` | Конвертация валют | exchangerate-api.com |

### 3. Интеграция
Демонстрация работы fine-tuned модели с вызовом инструментов через LangChain Agent.

---

## Быстрый старт

### Требования
- macOS M1/M2/M3 (Apple Silicon) или Linux с NVIDIA GPU
- Docker Desktop с поддержкой GPU (для Apple Silicon: Virtualization Framework)
- Python 3.11+ (для локальной разработки)
- uv (менеджер пакетов)

### Архитектура

Теперь приложение работает в **гибридном режиме**:
- **Docker**: только контейнеры Ollama с моделями (ollama-base и ollama-lora)
- **Локально**: Python приложение (API, обучение, импорт модели)

```
┌─────────────────┐     ┌────────────┐     ┌────────────┐
│  FastAPI App    │────▶│  Ollama    │     │  Ollama    │
│  (локально)     │     │   Base     │     │   LoRA     │
│  port 8000      │     │ port 11434 │     │ port 11435 │
└─────────────────┘     └────────────┘     └────────────┘
                              ▲                  ▲
                              │                  │
                         Docker             Docker
                        Container          Container
```

### 1. Запуск Ollama через Docker Compose

```bash
# Запуск только контейнеров Ollama
docker compose up -d

# Проверка статуса
docker compose ps
```

### 2. Загрузка базовой модели в ollama-base (выполняется один раз)

```bash
docker exec ollama-base ollama pull qwen2.5:3b
```

### 3. Установка зависимостей для локального приложения

```bash
# Установка uv (если не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Синхронизация зависимостей
uv sync

# Установка дополнительных пакетов для обучения
uv pip install trl accelerate bitsandbytes
```

### 4. Запуск API сервера локально

```bash
# Запуск API сервера
uv run python -m src.api

# Или через uvicorn напрямую
uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Проверка работоспособности

```bash
# Health check
curl http://localhost:8000/health

# Проверка статуса моделей
curl http://localhost:8000/models/status
```

---

## Сценарии использования и тестирования

### 1. Прямой запрос к модели (без инструментов)

Обычный диалог, модель отвечает напрямую:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Расскажи короткую историю про космос на русском языке."}'
```

**Ожидаемый результат:** `source: "direct"`

### 2. Тест инструмента: Курс валют

Агент распознаёт намерение получить курс валюты и вызывает инструмент:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Какой сейчас курс доллара (USD) к рублю (RUB)?"}'
```

**Ожидаемый результат:** `source: "agent"`, ответ содержит актуальный курс

### 3. Тест инструмента: Перевод текста

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Переведи фразу \"Hello world\" на русский язык."}'
```

**Ожидаемый результат:** `source: "agent"`

### 4. Тест инструмента: Определение языка

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "На каком языке написано: \"Guten Tag\"?"}'
```

**Ожидаемый результат:** `source: "agent"`, определение немецкого языка

### 5. Сложный запрос с контекстом

Проверка работы с историей диалога:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "А сколько будет стоить 50 евро в рублях по этому курсу?",
    "history": [
      {"role": "user", "content": "Какой курс EUR к RUB?"},
      {"role": "assistant", "content": "Курс 1 EUR = 90 RUB (пример)"}
    ]
  }'
```

### 6. Локальное тестирование инструментов (без Docker)

```bash
# Установка зависимостей через uv
uv sync

# Запуск тестов инструментов
uv run python -m src.test_tools
```

### 7. Тестирование обучения LoRA

```bash
# Подготовка окружения для обучения
uv pip install trl accelerate bitsandbytes

# Запуск обучения на небольшом наборе данных
uv run python -m src.train --max-samples 500 --epochs 1 --output ./test_adapter
```

### 8. Сравнение моделей через API (Base vs LoRA)

**Параллельный запрос к обеим моделям:**

```bash
curl -X POST http://localhost:8000/chat/compare \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Переведи фразу '\''Hello world'\'' на французский."
  }' | jq .
```

**Ответ содержит оба ответа одновременно:**

```json
{
  "base_response": "...",
  "lora_response": "...",
  "base_model": "qwen2.5:3b",
  "lora_model": "qwen-lora",
  "base_latency_ms": 1234,
  "lora_latency_ms": 1156
}
```

### 9. Запрос только к базовой модели

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Привет! Как дела?"}'
```

### 10. Запрос только к LoRA модели

```bash
curl -X POST http://localhost:8000/chat/lora \
  -H "Content-Type: application/json" \
  -d '{"message": "Расскажи историю про кота на немецком языке."}'
```

### 11. Автоматическое сравнение через скрипт

```bash
# Запуск скрипта сравнения (проверяет статус моделей и отправляет тестовые запросы)
python src/test_comparison.py
```

Скрипт автоматически:
- Проверяет доступность обеих моделей
- Отправляет 6 тестовых запросов (перевод, валюта, язык, диалог, креатив)
- Показывает разницу в длине ответов и скорости генерации
- Выводит статистику по latency

---

## Как используется LoRA и Tools: подробное объяснение

### Архитектура взаимодействия

Важно понимать разницу между **дообучением весов (LoRA)** и **использованием инструментов (Tools)**:

1. **LoRA (Low-Rank Adaptation)**:
   - Изменяет веса модели, чтобы она лучше понимала инструкции и формат диалога.
   - Обучается на датасете OpenAssistant oasst1.
   - Результат: новые веса (адаптеры), которые нужно "слить" с базовой моделью.

2. **Tools (LangChain)**:
   - Это внешние функции (API), которые модель вызывает по необходимости.
   - Не изменяют веса модели, а расширяют её возможности через контекст.
   - Модель учится *понимать*, когда нужно вызвать инструмент, благодаря промпту и дообучению.

### В какой момент появляется дообучение?

**Текущая реализация (Docker + Ollama с двумя контейнерами):**

1. Вы запускаете `docker compose up`, который поднимает:
   - `ollama-base` с **базовой** моделью `qwen2.5:3b` (порт 11434)
   - `ollama-lora` пустой контейнер для будущей модели (порт 11435)

2. Скрипт `src/train.py` обучает адаптеры LoRA и сохраняет их в папку `./lora_adapters`.

3. **Критический момент**: Запуск `src/merge_and_export.py`:
   - Сливаем базовую модель и адаптеры LoRA
   - **Автоматически** импортируем результат в контейнер `ollama-lora`
   - Модель становится доступна как `qwen-lora`

4. API `/chat/compare` отправляет запросы **параллельно** в оба контейнера.

**Преимущества dual-контейнеров:**
- Не нужно перезапускать сервисы для сравнения
- Мгновенная визуализация разницы "До" и "После"
- Оба ответа приходят одновременно через один API вызов

### Сценарий 12: Обучение LoRA и импорт модели в ollama-lora

```bash
# 1. Обучение LoRA (локально)
uv run python -m src.train --model Qwen/Qwen2.5-3B-Instruct --epochs 3 --batch-size 4

# 2. Слияние весов и АВТОМАТИЧЕСКИЙ импорт в ollama-lora
uv run python -m src.merge_and_export

# После успешного выполнения:
# - Модель qwen-lora доступна в http://localhost:11435
# - API /chat/compare готов к сравнению
```

**Что делает скрипт автоматически:**
1. ✅ Сливаем веса (Base + LoRA) → `./merged_model`
2. ✅ Копируем в контейнер `ollama-lora`
3. ✅ Создаем Modelfile
4. ✅ Выполняем `ollama create qwen-lora`
5. ✅ Модель готова к использованию!

### Сценарий 13: Проверка статуса моделей перед сравнением

```bash
# Проверка доступности обеих моделей
curl http://localhost:8000/models/status | jq .
```

**Ожидаемый ответ:**
```json
{
  "base": {
    "available": true,
    "url": "http://localhost:11434",
    "models": ["qwen2.5:3b"],
    "target_model_present": true
  },
  "lora": {
    "available": true,
    "url": "http://localhost:11435",
    "models": ["qwen-lora"],
    "target_model_present": true
  }
}
```

---

## API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/health` | Проверка здоровья сервиса |
| GET | `/models/status` | Статус обеих моделей (base и lora) |
| POST | `/chat` | Запрос к базовой модели |
| POST | `/chat/lora` | Запрос к LoRA модели |
| POST | `/chat/compare` | **Параллельный запрос к обеим моделям** |

### Формат запроса `/chat`, `/chat/lora`

```json
{
  "message": "Текст сообщения",
  "history": []
}
```

### Формат запроса `/chat/compare`

```json
{
  "message": "Текст сообщения"
}
```

### Формат ответа (одиночный запрос)

```json
{
  "response": "Ответ модели",
  "source": "direct или agent",
  "model": "qwen2.5:3b или qwen-lora",
  "latency_ms": 1234.56
}
```

### Формат ответа (сравнение)

```json
{
  "base_response": "Ответ базовой модели",
  "lora_response": "Ответ LoRA модели",
  "base_model": "qwen2.5:3b",
  "lora_model": "qwen-lora",
  "base_latency_ms": 1234.56,
  "lora_latency_ms": 1156.78
}
```

---

## Разработка

### Структура проекта

```
/workspace
├── docker-compose.yml      # Конфигурация Docker Compose (2x Ollama)
├── Dockerfile              # Образ Python приложения с uv
├── pyproject.toml          # Зависимости Python
├── README.md               # Документация
└── src/
    ├── api.py              # FastAPI приложение (dual-model API)
    ├── tools.py            # LangChain инструменты (3 tools)
    ├── train.py            # Скрипт обучения LoRA (PEFT, 4-bit)
    ├── merge_and_export.py # Слияние LoRA + импорт в Ollama
    ├── test_comparison.py  # Скрипт сравнения моделей
    ├── inference.py        # Инференс с LoRA адаптером
    └── test_tools.py       # Тесты инструментов
```

### Локальный запуск приложения

```bash
# Установка uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Синхронизация зависимостей
uv sync

# Запуск API сервера
uv run python -m src.api

# Или через uvicorn напрямую
uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Добавление новых инструментов

1. Создайте класс инструмента в `src/tools.py`:

```python
class MyNewTool:
    name = "my_new_tool"
    description = "Описание инструмента"
    
    def _run(self, *args, **kwargs) -> str:
        # Логика инструмента
        return "result"
```

2. Добавьте инструмент в `get_all_tools()` и создайте LangChain wrapper.

3. Обновите триггерные слова в `src/api.py` при необходимости.

---

## Troubleshooting

### Модель не загружается в Ollama

```bash
# Проверьте доступность Ollama
docker exec ollama-server ollama list

# Переустановите модель
docker exec ollama-server ollama rm qwen2.5:3b
docker exec ollama-server ollama pull qwen2.5:3b
```

### Ошибки памяти при обучении LoRA

- Уменьшите `--batch-size` до 1 или 2
- Уменьшите `--max-samples`
- Используйте `--no-4bit` только если у вас > 24GB GPU памяти

### Проблемы с Apple GPU в Docker

Убедитесь, что в Docker Desktop включена поддержка Virtualization Framework:
- Settings → Resources → Advanced → Enable Virtualization Framework

**Примечание:** В новой архитектуре контейнеры Ollama работают без прямого доступа к GPU, 
используя CPU. Для обучения модели с GPU ускорением запускайте скрипт `train.py` локально 
на машине с NVIDIA GPU или используйте облачные сервисы (Google Colab, Kaggle).

---

## Лицензия

MIT License
