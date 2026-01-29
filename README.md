# AI Screenwriter Assistant

Помощник для сценаристов на базе Gemini 2.5 Flash с использованием архитектуры RAG (Retrieval-Augmented Generation).

Проект позволяет консультироваться по теории сценарного мастерства, опираясь на загруженные учебники (например, Джозеф Кэмпбелл), и генерировать сцены в правильном формате.

## Основные возможности
- Smart Consultation (RAG): ИИ ищет ответы в базе знаний (PDF-данных) и выдает советы со ссылками на первоисточники.
- Scene Generation: Генерация сценариев по заданным параметрам (жанр, персонажи, тон).
- Vector Database: Использование ChromaDB для хранения и поиска смысловых векторов.

## Технологический стек
- Python 3.11+
- FastAPI (Backend framework)
- Google GenAI SDK (Gemini 2.5 Flash)
- ChromaDB (Vector Database)
- PyMuPDF (PDF processing)
- Docker & Docker Compose

## Как запустить

### 1. Клонирование репозитория
```bash
git clone [https://github.com/Kate-vsn/ai-screenwriter-assistant.git](https://github.com/Kate-vsn/ai-screenwriter-assistant.git)
cd ai-screenwriter-assistant
```

### 2. Запуск проекта (Docker)

Проект полностью контейнеризирован. Вам не нужно устанавливать Python локально.

#### 1. Подготовка
Создайте файл `.env` в корневой папке и добавьте ваш ключ:
```ini
LLM_API_KEY="ваш_ключ_из_google_ai_studio"
LLM_MODEL="gemini-2.5-flash"
```
#### 2. Выполните команды:
```
docker-compose up --build
docker exec -it screenwriter_api python -m src.rag.ingest
```

После запуска Swagger UI (интерфейс для тестов) будет доступен по адресу: http://localhost:8000/docs

