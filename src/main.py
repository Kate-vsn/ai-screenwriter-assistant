from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from src.config import settings
from src.rag.service import search_context 

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

class SceneRequest(BaseModel):
    genre: str
    characters: str
    plot_outline: str
    tone: str

class ConsultRequest(BaseModel):
    question: str

class AIResponse(BaseModel):
    response: str
    context_used: bool

client = genai.Client(api_key=settings.LLM_API_KEY)

@app.get("/")
async def root():
    return {"status": "ok", "service": "AI Screenwriter + RAG"}

@app.post("/generate/scene")
async def generate_scene(request: SceneRequest):
    """
    Генерирует черновик сцены в сценарном формате, используя LLM.
    """
    
    system_instruction = """
    Ты — профессиональный голливудский сценарист.
    Твоя задача — написать сцену на основе вводных данных.
    
    ФОРМАТ ВЫВОДА (СТРОГО СОБЛЮДАЙ FOUNTAIN/СЦЕНАРНЫЙ ФОРМАТ):
    1. Заголовки сцен (Sluglines) пиши КАПСОМ: "ИНТ. КОМНАТА - ДЕНЬ" или "EXT. STREET - NIGHT".
    2. Имена персонажей перед диалогом пиши КАПСОМ.
    3. Ремарки (parentheticals) пиши в скобках на отдельной строке под именем.
    4. Описание действия (Action lines) пиши обычным текстом.
    5. НЕ пиши никаких вступлений типа "Вот ваша сцена" или "Черновик готов". Начинай сразу с заголовка сцены.
    6. Пиши ЧИСТЫМ ТЕКСТОМ. Строго запрещены HTML теги (<center> и т.д.).
    СТИЛЬ:
    - Диалоги должны быть живыми, с подтекстом.
    - Избегай клише.
    - Соблюдай заданный пользователем тон.
    Сам ВЫВОД (СТРОГИЙ FOUNTAIN):
    1. СЦЕНАРНЫЙ БЛОК:
       [ПУСТАЯ СТРОКА]
       ИМЯ ПЕРСОНАЖА (Всегда на русском, КАПСОМ)
       (ремарка - если есть, в скобках, с маленькой буквы)
       Текст диалога
       [ПУСТАЯ СТРОКА]
    
    2. ВАЖНЕЙШИЕ ПРАВИЛА:
       - НИКОГДА не пиши Имя и Диалог в одну строку.
       - НИКОГДА не пиши Ремарку и Диалог в одну строку.
       - Между именем героя и его репликой ОБЯЗАТЕЛЬНО должен быть перенос строки.
       - Имена персонажей пиши только на РУССКОМ языке (МАРК, а не MARK).
       
    3. ОПИСАНИЕ ДЕЙСТВИЯ:
       Пиши обычными абзацами. Оставляй пустую строку перед заголовком сцены.

    ПРИМЕР ТОЧНОГО ВЫПОЛНЕНИЯ:
    
    МАРК
    (устало)
    Слава богу.

    ЛЕНА
    (не отрываясь)
    Как всегда.
    """

    full_prompt = f"""
    Напиши сцену по следующим параметрам:
    
    Жанр: {request.genre}
    Персонажи: {request.characters}
    Тон: {request.tone}
    
    Сюжетное описание сцены: 
    {request.plot_outline}
    """

    try:
        resp = client.models.generate_content(
            model=settings.LLM_MODEL, 
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7
            ) 
        )
        
        return {"scene_script": resp.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Generation Error: {str(e)}")

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