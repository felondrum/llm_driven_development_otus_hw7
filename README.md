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

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   FastAPI App   │────▶│  LangChain Agent │────▶│  External Tools │
│   (port 8000)   │     │  (Zero-shot ReAct)│     │  (Currency, etc)│
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│    Ollama       │
│  qwen2.5:3b     │
│  (port 11434)   │
└─────────────────┘
```

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
- macOS M1/M2/M3 (Apple Silicon)
- Docker Desktop с поддержкой GPU
- Python 3.11+ (для локальной разработки)
- uv (менеджер пакетов)

### 1. Запуск сервисов через Docker Compose

```bash
# Запуск всех сервисов
docker compose up -d

# Проверка статуса
docker compose ps
```

### 2. Загрузка модели Ollama (выполняется один раз)

```bash
docker exec -it ollama-server ollama pull qwen2.5:3b
```

### 3. Проверка работоспособности

```bash
# Health check
curl http://localhost:8000/health

# Ожидается: {"status":"ok","model":"qwen2.5:3b"}
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

---

## API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/health` | Проверка здоровья сервиса |
| POST | `/chat` | Отправка сообщения модели |

### Формат запроса `/chat`

```json
{
  "message": "Текст сообщения",
  "history": [
    {"role": "user", "content": "Предыдущее сообщение"},
    {"role": "assistant", "content": "Предыдущий ответ"}
  ]
}
```

### Формат ответа

```json
{
  "response": "Ответ модели",
  "source": "direct или agent"
}
```

---

## Разработка

### Структура проекта

```
/workspace
├── docker-compose.yml      # Конфигурация Docker Compose
├── Dockerfile              # Образ Python приложения
├── pyproject.toml          # Зависимости Python
├── README.md               # Документация
└── src/
    ├── api.py              # FastAPI приложение
    ├── tools.py            # LangChain инструменты
    ├── train.py            # Скрипт обучения LoRA
    ├── inference.py        # Инференс с LoRA адаптером
    └── test_tools.py       # Тесты инструментов
```

### Локальный запуск (без Docker)

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

Убедитесь, что в Docker Desktop включена поддержка GPU:
- Settings → Resources → GPU → Enable Apple GPU acceleration

---

## Лицензия

MIT License
