import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import settings

DATA_DIR = "data"
PERSIST_DIRECTORY = "chromadb_storage"

def ingest_docs():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.LLM_API_KEY
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="screenwriter_knowledge"
    )

    categories = ["theory", "psychology", "fiction"]
    
    for cat in categories:
        cat_path = os.path.join(DATA_DIR, cat)
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
            continue

        print(f"Обработка категории: {cat.upper()}")
        files = [f for f in os.listdir(cat_path) if f.endswith('.pdf')]

        for file_name in files:
            file_path = os.path.join(cat_path, file_name)
            
            existing = vectorstore.get(where={"source": file_path})
            if len(existing['ids']) > 0:
                print(f"Пропуск: {file_name}")
                continue

            print(f"Индексация [{cat.upper()}]: {file_name}")
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["category"] = cat
                doc.metadata["source"] = file_path 

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            vectorstore.add_documents(chunks)
            print(f"Добавлено {len(chunks)} чанков")

if __name__ == "__main__":
    ingest_docs()