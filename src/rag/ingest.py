import os
import chromadb
from chromadb.utils import embedding_functions
from google import genai
import fitz 
from src.config import settings

chroma_client = chromadb.PersistentClient(path="chromadb_storage")

class GoogleGenAIEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model="text-embedding-004", 
            contents=input
        )
        return [embedding.values for embedding in response.embeddings]

collection = chroma_client.get_or_create_collection(
    name="screenwriting_knowledge",
    embedding_function=GoogleGenAIEmbeddingFunction(api_key=settings.LLM_API_KEY)
)

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        print(f"Не удалось прочитать {pdf_path}: {e}")
        return ""

def ingest_data():
    data_folder = "data"
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        return

    files = [f for f in os.listdir(data_folder) if f.endswith('.pdf') or f.endswith('.txt')]
    
    if not files:
        print("В папке data пусто.")
        return

    print(f"Найдено файлов: {len(files)}")

    for filename in files:
        file_path = os.path.join(data_folder, filename)
        print(f"Обрабатываю: {filename}...")

        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        if not text:
            print("Пусто, пропускаю.")
            continue

        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        if not chunks:
            continue

        print(f"Всего фрагментов: {len(chunks)}")

        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_id": i} for i in range(len(chunks))]
        
        batch_size = 50
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            try:
                collection.add(
                    documents=batch_chunks,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
                print(f"Загружено {i + len(batch_chunks)} из {len(chunks)}...")
            except Exception as e:
                print(f"Ошибка на пакете {i}: {e}")

if __name__ == "__main__":
    ingest_data()