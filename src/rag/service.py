import chromadb
from chromadb.utils import embedding_functions
from google import genai
from src.config import settings

class GoogleGenAIEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model="text-embedding-004", 
            contents=input
        )
        return [embedding.values for embedding in response.embeddings]

chroma_client = chromadb.PersistentClient(path="chromadb_storage")
collection = chroma_client.get_collection(
    name="screenwriting_knowledge",
    embedding_function=GoogleGenAIEmbeddingFunction(api_key=settings.LLM_API_KEY)
)

def search_context(query: str, n_results: int = 3) -> str:
    """
    Ищет в базе знаний фрагменты, похожие на запрос.
    Возвращает их одной строкой текста.
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        documents = results['documents'][0]
        
        context_text = "\n\n--- ФРАГМЕНТ ИЗ КНИГИ ---\n".join(documents)
        return context_text
        
    except Exception as e:
        print(f"Ошибка поиска в базе: {e}")
        return ""