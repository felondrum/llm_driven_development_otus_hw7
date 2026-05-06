import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama
from langchain.tools import Tool

from tools import detect_language_tool, translate_text_tool, get_currency_rate_tool

app = FastAPI(title="Multilingual LLM Assistant with Ollama")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama-server:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b")

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    source: str  # 'direct' или 'agent'

def get_llm():
    return Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_HOST,
        temperature=0.7
    )

def create_agent():
    llm = get_llm()
    tools = [
        detect_language_tool,
        translate_text_tool,
        get_currency_rate_tool
    ]
    
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

@app.on_event("startup")
async def startup_event():
    print(f"Connecting to Ollama at {OLLAMA_HOST}")
    print(f"Using model: {MODEL_NAME}")
    # Проверка подключения можно добавить здесь

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Простая эвристика: если запрос про перевод или язык - используем агента
        lower_msg = request.message.lower()
        trigger_words = ["translate", "переведи", "language", "язык", "currency", "курс", "rate"]
        
        if any(word in lower_msg for word in trigger_words):
            agent = create_agent()
            response = agent.run(request.message)
            return ChatResponse(response=response, source="agent")
        else:
            # Прямой запрос к модели без инструментов
            llm = get_llm()
            response = llm.invoke(request.message)
            return ChatResponse(response=response, source="direct")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)