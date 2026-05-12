import os
import json
import httpx
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.llms import Ollama
from langchain_classic.tools import Tool
from langchain_core.prompts import PromptTemplate

from src.tools import detect_language_tool, translate_text_tool, get_currency_rate_tool

app = FastAPI(title="Multilingual LLM Assistant with Ollama - Dual Model Comparison")

# URLs для двух экземпляров Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LORA_URL = os.getenv("OLLAMA_LORA_URL", "http://localhost:11435")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:1.5b")
LORA_MODEL_NAME = os.getenv("LORA_MODEL_NAME", "qwen-lora")

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    source: str  # 'direct', 'agent', 'base', 'lora'
    model: str
    latency_ms: Optional[float] = None

class ComparisonResponse(BaseModel):
    base_response: str
    lora_response: str
    base_model: str
    lora_model: str
    base_latency_ms: float
    lora_latency_ms: float

def get_llm(base_url: str, model: str):
    return Ollama(
        model=model,
        base_url=base_url,
        temperature=0.7
    )

def create_agent(base_url: str, model: str):
    llm = get_llm(base_url, model)
    tools = [
        detect_language_tool,
        translate_text_tool,
        get_currency_rate_tool
    ]
    
   # ReAct prompt template for the agent
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""
        )

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

async def query_ollama(base_url: str, model: str, prompt: str, use_agent: bool = False) -> tuple[str, float]:
    """Запрос к Ollama с замером времени"""
    import time
    start_time = time.time()
    
    try:
        if use_agent:
            agent = create_agent(base_url, model)
            response = agent.invoke({"input": prompt})
            # AgentExecutor returns a dict with 'output' key
            response = response.get("output", str(response))
        else:
            llm = get_llm(base_url, model)
            response = llm.invoke(prompt)
        
        latency_ms = (time.time() - start_time) * 1000
        return response, latency_ms
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        raise HTTPException(status_code=500, detail=f"Error querying {model}: {str(e)}")

@app.on_event("startup")
async def startup_event():
    print(f"Connecting to Ollama Base at {OLLAMA_BASE_URL}")
    print(f"Connecting to Ollama LoRA at {OLLAMA_LORA_URL}")
    print(f"Base model: {MODEL_NAME}")
    print(f"LoRA model: {LORA_MODEL_NAME}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Чат с базовой моделью (по умолчанию)"""
    try:
        lower_msg = request.message.lower()
        trigger_words = ["translate", "переведи", "language", "язык", "currency", "курс", "rate"]
        
        use_agent = any(word in lower_msg for word in trigger_words)
        response, latency_ms = await query_ollama(OLLAMA_BASE_URL, MODEL_NAME, request.message, use_agent)
        
        return ChatResponse(
            response=response, 
            source="agent" if use_agent else "direct",
            model=MODEL_NAME,
            latency_ms=latency_ms
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/lora", response_model=ChatResponse)
async def chat_lora(request: ChatRequest):
    """Чат с дообученной LoRA моделью"""
    try:
        lower_msg = request.message.lower()
        trigger_words = ["translate", "переведи", "language", "язык", "currency", "курс", "rate"]
        
        use_agent = any(word in lower_msg for word in trigger_words)
        response, latency_ms = await query_ollama(OLLAMA_LORA_URL, LORA_MODEL_NAME, request.message, use_agent)
        
        return ChatResponse(
            response=response, 
            source="agent" if use_agent else "direct",
            model=LORA_MODEL_NAME,
            latency_ms=latency_ms
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/compare", response_model=ComparisonResponse)
async def chat_compare(request: ChatRequest):
    """Сравнительный ответ от обеих моделей одновременно"""
    try:
        lower_msg = request.message.lower()
        trigger_words = ["translate", "переведи", "language", "язык", "currency", "курс", "rate"]
        use_agent = any(word in lower_msg for word in trigger_words)
        
        # Параллельные запросы к обеим моделям
        import asyncio
        base_task = query_ollama(OLLAMA_BASE_URL, MODEL_NAME, request.message, use_agent)
        lora_task = query_ollama(OLLAMA_LORA_URL, LORA_MODEL_NAME, request.message, use_agent)
        
        base_response, base_latency = await base_task
        lora_response, lora_latency = await lora_task
        
        return ComparisonResponse(
            base_response=base_response,
            lora_response=lora_response,
            base_model=MODEL_NAME,
            lora_model=LORA_MODEL_NAME,
            base_latency_ms=base_latency,
            lora_latency_ms=lora_latency
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "base_model": MODEL_NAME,
        "lora_model": LORA_MODEL_NAME,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_lora_url": OLLAMA_LORA_URL
    }

@app.get("/models/status")
async def models_status():
    """Проверка доступности обеих моделей"""
    import httpx
    
    status = {}
    
    # Проверка базовой модели
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                status["base"] = {
                    "available": True,
                    "url": OLLAMA_BASE_URL,
                    "models": [m["name"] for m in models],
                    "target_model_present": any(MODEL_NAME in m["name"] for m in models)
                }
            else:
                status["base"] = {"available": False, "error": f"Status {response.status_code}"}
    except Exception as e:
        status["base"] = {"available": False, "error": str(e)}
    
    # Проверка LoRA модели
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_LORA_URL}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                status["lora"] = {
                    "available": True,
                    "url": OLLAMA_LORA_URL,
                    "models": [m["name"] for m in models],
                    "target_model_present": any(LORA_MODEL_NAME in m["name"] for m in models)
                }
            else:
                status["lora"] = {"available": False, "error": f"Status {response.status_code}"}
    except Exception as e:
        status["lora"] = {"available": False, "error": str(e)}
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)