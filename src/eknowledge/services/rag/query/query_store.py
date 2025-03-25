import logging

from llama_index.core import Settings, PropertyGraphIndex, VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
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
    # 自定义 QA 提示模板，改成中文
    # https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/custom_prompt_synthesizer/
    qa_template = PromptTemplate(
        "以下是上下文信息：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "请基于以上上下文信息（而非先前的知识）回答以下查询。\n"
        "查询: {query_str}\n"
        "答案:"
    )

    refiner_template = PromptTemplate(
        "原始问题如下：{query_str}\n"
        "我们已有一个现有答案：{existing_answer}\n"
        "现在我们可以根据以下的更多上下文信息（如果有需要）来完善现有答案。\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "基于新的上下文信息，完善现有答案以更好地回答问题。\n"
        "如果上下文信息无用，则保持原答案不变。\n"
        "优化后的答案：")

    # 使用自定义模板创建响应合成器
    response_synthesizer = get_response_synthesizer(
        # 解释response_mode https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes/
        response_mode=ResponseMode.COMPACT,
        text_qa_template=qa_template,
        refine_template=refiner_template,
        # refine模块能够过滤掉与所问问题无关的任何输入节点
        structured_answer_filtering=True,
    )

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
    logger.info(response)
