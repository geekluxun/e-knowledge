from llama_index.core import Settings, PropertyGraphIndex, VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from eknowledge.model import getLLM, getEmbeddings
from eknowledge.store.storage import get_storage_context


def query_from_vector_store(vector_store, prompt: str):
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine()
    return query_engine.query(prompt)


def query_from_graph_store(graph_store, prompt: str):
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        llm=Settings.llm,
        embed_model=Settings.embed_model,
    )
    query_engine = index.as_query_engine(include_text=True)
    return query_engine.query(prompt)


def hybrid_query(storage_context: StorageContext, prompt: str):
    index = VectorStoreIndex.from_vector_store(storage_context.vector_store)

    retriever = QueryFusionRetriever(
        [
            index.as_retriever(similarity_top_k=2),
            BM25Retriever.from_defaults(
                docstore=storage_context.docstore, similarity_top_k=2
            ),
        ],
        num_queries=1,
        use_async=True,
    )
    response_synthesizer = get_response_synthesizer()
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    return query_engine.query(prompt)


if __name__ == '__main__':
    storage_context = get_storage_context()
    Settings.llm = getLLM()
    Settings.embed_model = getEmbeddings()

    response = query_from_graph_store(storage_context.property_graph_store, "MCopilot是什么？")
    print(response)

    response = query_from_vector_store(storage_context.vector_store, "MCopilot是什么？")
    print(response)

    response = hybrid_query(storage_context, "MCopilot是什么？")
    print(response)
