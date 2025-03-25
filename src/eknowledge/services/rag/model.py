from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


def getLLM():
    llm = Ollama(
        model="qwen2:7b-instruct",
        temperature=0,
        context_window=16384,
    )
    return llm


def getEmbeddings():
    embeddings = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )
    return embeddings
