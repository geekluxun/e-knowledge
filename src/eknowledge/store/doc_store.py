from llama_index.storage.docstore.redis import RedisDocumentStore


def get_doc_store():
    store = RedisDocumentStore.from_host_and_port(
        "localhost", 6379
    )
    return store
