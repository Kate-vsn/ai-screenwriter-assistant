from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from src.config import settings
from src.rag.service import search_context 

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

class ScenarioRequest(BaseModel):
    genre: str
    characters: str
    plot_point: str
    tone: str = "dramatic"

class ConsultRequest(BaseModel):
    question: str

class AIResponse(BaseModel):
    response: str
    context_used: bool

client = genai.Client(api_key=settings.LLM_API_KEY)

@app.get("/")
async def root():
    return {"status": "ok", "service": "AI Screenwriter + RAG"}

@app.post("/generate/scene", response_model=AIResponse)
async def generate_scene(request: ScenarioRequest):
    prompt = f"""
    Напиши сцену.
    Жанр: {request.genre}. Персонажи: {request.characters}.
    Ситуация: {request.plot_point}. Тон: {request.tone}.
    """
    try:
        resp = client.models.generate_content(model=settings.LLM_MODEL, contents=prompt)
        return AIResponse(response=resp.text, context_used=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consult", response_model=AIResponse)
async def consult_guru(request: ConsultRequest):
    """
    Отвечает на вопросы по теории сценария, используя загруженные книги.
    """
    found_context = search_context(request.question, n_results=3)
    system_instruction = "Ты эксперт по сценарному мастерству. Используй ТОЛЬКО предоставленный ниже контекст из книг, чтобы ответить на вопрос. Если в контексте нет ответа, так и скажи."
    
    full_prompt = f"""
    Вопрос студента: {request.question}
    
    Найденная информация из учебников:
    {found_context}
    
    Дай развернутый совет, опираясь на эти материалы.
    """

    try:
        resp = client.models.generate_content(
            model=settings.LLM_MODEL, 
            contents=full_prompt,
            config=types.GenerateContentConfig(system_instruction=system_instruction) # Добавляем системную инструкцию
        )
        return AIResponse(response=resp.text, context_used=bool(found_context))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")