FROM python:3.11-slim

WORKDIR /app

# Установка uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Копирование файлов зависимостей
COPY pyproject.toml uv.lock ./

# Установка зависимостей в виртуальное окружение
RUN uv venv && \
    uv pip install -e .

# Копирование исходного кода
COPY src/ ./src/

# Экспозиция порта
EXPOSE 8000

# Запуск приложения
CMD ["uv", "run", "python", "src/api.py"]