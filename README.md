# Multilingual Dialog Assistant with LoRA and LangChain Tools

## Overview
This project implements a multilingual dialog assistant using:
- **PEFT (LoRA)** for parameter-efficient fine-tuning on 35 languages
- **LangChain** for external API tools (translation, language detection, currency conversion)
- **OpenAssistant oasst1** dataset with 84.4k dialogs
- **Qwen2** as the base model

## Project Structure
```
/workspace
├── pyproject.toml          # Python project configuration
├── docker-compose.yml      # Docker Compose for model serving
├── Dockerfile              # Docker image for inference server
├── src/
│   ├── __init__.py
│   ├── train.py            # LoRA fine-tuning script
│   ├── tools.py            # LangChain tools implementation
│   ├── inference.py        # Inference with tool integration
│   ├── api.py              # FastAPI REST API server
│   ├── test_tools.py       # Tool tests
│   └── models/             # Model storage
└── README.md               # This file
```

## Requirements
- Mac M1 (Apple Silicon) or compatible system
- Docker Desktop with Docker Compose
- Python 3.10+ with uv package manager
- At least 16GB RAM (32GB recommended for training)

## Quick Start

### 1. Install Dependencies
```bash
# Install core dependencies (FastAPI, LangChain tools)
uv sync

# Install ML dependencies for training/inference (optional, large download)
uv sync --extra ml
```

### 2. Start Inference Server (Docker)
```bash
docker compose up -d
```

The server will be available at `http://localhost:8000`

### 3. Run Fine-tuning (requires ML extras)
```bash
uv run python src/train.py
```

### 4. Run Inference with Tools (requires ML extras)
```bash
# Test tools only (no model required)
uv run python src/test_tools.py

# Run full inference with model
uv run python src/inference.py --test
uv run python src/inference.py --interactive
```

### 5. Run API Server locally
```bash
uv run python src/api.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/chat` | POST | Chat with assistant |
| `/detect-language` | POST | Detect text language |
| `/translate` | POST | Translate text |
| `/convert-currency` | POST | Convert currencies |

### Example API Usage

```bash
# Detect language
curl -X POST http://localhost:8000/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text": "Привет, как дела?"}'

# Currency conversion
curl -X POST http://localhost:8000/convert-currency \
  -H "Content-Type: application/json" \
  -d '{"amount": 100, "from_currency": "USD", "to_currency": "EUR"}'
```

## Architecture

### Fine-tuning Pipeline
1. Load OpenAssistant oasst1 dataset from HuggingFace
2. Tokenize conversations for Qwen2 model
3. Apply LoRA adapters for parameter-efficient training
4. Train on multi-lingual dialogs (35+ languages)
5. Evaluate against base model

### LangChain Tools
- **detect_language**: Identify input language using character analysis
- **translate_text**: Translate between languages (mock implementation)
- **currency_converter**: Convert currencies using real-time rates from exchangerate-api.com

## Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2-1.5B-Instruct` | Base model to use |
| `LORA_ADAPTER_PATH` | `None` | Path to LoRA adapter |
| `DEVICE` | `cpu` | Device for inference |
| `MAX_LENGTH` | `512` | Maximum sequence length |

## License
MIT
