import logging

from llama_index.core import Settings, PropertyGraphIndex, VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.retrievers.bm25 import BM25Retriever

from eknowledge.services.rag import init_rag_settings
from eknowledge.services.rag.context import MyStorageContext

logger = logging.getLogger(__name__)

def query_from_vector_store(vector_store, metadata_filter, prompt: str):
    index = VectorStoreIndex.from_vector_store(vector_store)
    # Metadata过滤，精确匹配相关文档，可以增加用户部门权限等
    filters_list = [
        ExactMatchFilter(key=k, value=v)
        for k, v in metadata_filter.items()
    ]
    filters = MetadataFilters(filters=filters_list)
    query_engine = index.as_query_engine(filters=filters, similarity_top_k=5)
    response = query_engine.query(prompt)
    return response


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
    storage_context = MyStorageContext.storageContext()
    init_rag_settings()

    # 定义根据metadata过滤
    metadata_filter = {
        "department": "qa",
    }
    response = query_from_graph_store(storage_context.property_graph_store, "MCopilot是什么？")
    logger(response)

    response = query_from_vector_store(storage_context.vector_store, metadata_filter=metadata_filter, prompt="能力模型是指什么？")
    logger(response)

    response = hybrid_query(storage_context, "MCopilot是什么？")
    logger(response)
