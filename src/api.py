"""
FastAPI Server for Model Inference

This module provides a REST API for the multilingual assistant.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os

app = FastAPI(
    title="Multilingual Assistant API",
    description="API for multilingual dialog assistant with LoRA and LangChain tools",
    version="0.1.0"
)

# Global model instance (loaded on first request)
_model = None


def get_model():
    """Lazy load the model."""
    global _model
    if _model is None:
        from src.inference import MultilingualAssistant
        
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2-1.5B-Instruct")
        adapter_path = os.getenv("LORA_ADAPTER_PATH", None)
        device = os.getenv("DEVICE", "cpu")
        
        _model = MultilingualAssistant(
            base_model_name=model_name,
            lora_adapter_path=adapter_path,
            device=device,
        )
    
    return _model


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    max_length: int = 512


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    detected_language: Optional[str] = None


class LanguageDetectRequest(BaseModel):
    """Language detection request."""
    text: str


class LanguageDetectResponse(BaseModel):
    """Language detection response."""
    language: str


class TranslateRequest(BaseModel):
    """Translation request."""
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"


class TranslateResponse(BaseModel):
    """Translation response."""
    translated_text: str
    source_language: str
    target_language: str


class CurrencyConvertRequest(BaseModel):
    """Currency conversion request."""
    amount: float
    from_currency: str
    to_currency: str


class CurrencyConvertResponse(BaseModel):
    """Currency conversion response."""
    result: str
    amount: float
    from_currency: str
    to_currency: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multilingual Assistant API",
        "endpoints": [
            "/chat",
            "/detect-language",
            "/translate",
            "/convert-currency",
            "/health"
        ]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the assistant."""
    try:
        model = get_model()
        
        # Detect language
        detected_lang = model.detect_language(request.message)
        
        # Generate response
        response = model.chat(request.message)
        
        return ChatResponse(
            response=response,
            detected_language=detected_lang
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-language", response_model=LanguageDetectResponse)
async def detect_language(request: LanguageDetectRequest):
    """Detect language of input text."""
    try:
        model = get_model()
        language = model.detect_language(request.text)
        return LanguageDetectResponse(language=language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """Translate text between languages."""
    try:
        model = get_model()
        
        # Auto-detect source language if needed
        source_lang = request.source_lang
        if source_lang == "auto":
            source_lang = model.detect_language(request.text)
        
        translated = model.translate(
            request.text,
            source_lang=source_lang,
            target_lang=request.target_lang
        )
        
        return TranslateResponse(
            translated_text=translated,
            source_language=source_lang,
            target_language=request.target_lang
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/convert-currency", response_model=CurrencyConvertResponse)
async def convert_currency(request: CurrencyConvertRequest):
    """Convert currency amount."""
    try:
        model = get_model()
        result = model.convert_currency(
            request.amount,
            request.from_currency,
            request.to_currency
        )
        
        return CurrencyConvertResponse(
            result=result,
            amount=request.amount,
            from_currency=request.from_currency,
            to_currency=request.to_currency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
