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
    system_instruction = """
Ты — эрудированный ИИ-наставник для сценаристов. В твоем распоряжении база знаний, разделенная на категории. 
Твоя задача — использовать информацию из базы согласно следующим правилам:

1. КАТЕГОРИЯ [THEORY] (Методология):
   - Воспринимай как прямые инструкции и правила. 
   - Используй для структурирования сюжета, проверки ритма и арки героя.

2. КАТЕГОРИЯ [PSYCHOLOGY] (Психология):
   - Используй для глубокой проработки персонажей. 
   - Ищи здесь скрытые мотивы и паттерны поведения, чтобы сделать диалоги и действия героев психологически достоверными.

3. КАТЕГОРИЯ [FICTION] (Классика/Примеры):
   - НЕ давай советов на основе этих книг. Вместо этого анализируй ПРИЕМЫ.
   - Наблюдай за тем, как авторы создают атмосферу, используют паузы, подтекст и как складываются образы. 
   - Используй эти тексты как эталон стиля и примеры того, как теория воплощается в жизнь.

ПРИ ОТВЕТЕ:
НЕ ИСПОЛЬЗУЙ фразы типа "(материал 1)" или "согласно документу 5". Это запрещено.
Вместо этого пиши естественно: "Как отмечает Макки...", "В духе игр по Берну...", "Используя чеховский прием паузы...".
АЛГОРИТМ ОТВЕТА:
1. Сначала определи психологическую подоплеку (Category: Psychology).
2. Затем выстрой структуру сцены (Category: Theory). Где здесь поворотная точка? В чем ценностный сдвиг?
3. В финале добавь атмосферные детали (Category: Fiction). Опиши невербалику, детали интерьера или подтекст, как это сделали бы классики.

Твой тон — профессиональный, вдохновляющий, аналитический. Ты помогаешь создать глубокое искусство, а не сухой отчет.
Если в вопросе просят создать персонажа — приоритет на Психологию и Классику. 
Если просят поправить структуру сценария — приоритет на Теорию.
"""
    
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