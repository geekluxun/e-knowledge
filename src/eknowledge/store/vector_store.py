from llama_index.vector_stores.milvus import MilvusVectorStore

dim = 1024


def get_vector_store():
    vector_store = MilvusVectorStore(
        uri="/Users/luxun/workspace/ai/mine/project/open/e-knowledge/db/milvus.db", overwrite=False, dim=dim
    )
    return vector_store
