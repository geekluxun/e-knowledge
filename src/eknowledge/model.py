from langchain_ollama import ChatOllama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def getLLM():
    llm = ChatOllama(
        model="qwen2:7b",
        temperature=0,
        # other params...
    )
    return llm


def getEmbeddings():
    embeddings = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )
    return embeddings
