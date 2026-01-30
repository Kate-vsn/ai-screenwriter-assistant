import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.config import settings

PERSIST_DIRECTORY = "chromadb_storage"
COLLECTION_NAME = "screenwriter_knowledge" 

def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.LLM_API_KEY
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    return vectorstore

def search_context(query: str, n_results: int = 5):
    """Ищет информацию в базе знаний"""
    try:
        vectorstore = get_vectorstore()
        
        results = vectorstore.similarity_search(query, k=n_results)
        
        if not results:
            return ""

        context_parts = []
        for doc in results:
            category = doc.metadata.get("category", "General")
            source_path = doc.metadata.get("source", "Unknown")
            source = source_path.split("/")[-1] if source_path else "Unknown"
            
            context_parts.append(
                f"[Source: {source} | Category: {category}]\n{doc.page_content}"
            )
            
        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        print(f"ОШИБКА В RAG SERVICE: {e}")
        return ""