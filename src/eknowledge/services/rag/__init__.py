from llama_index.core import Settings
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager

from eknowledge.services.rag.model import getLLM, getEmbeddings
from eknowledge.services.rag.store.doc_store import get_doc_store
from eknowledge.services.rag.store.graph_store import get_graph_store
from eknowledge.services.rag.store.vector_store import get_vector_store
from eknowledge.services.rag.trace.llm_logger import LLMLogger


def init_rag_settings():
    Settings.llm = getLLM()
    Settings.embed_model = getEmbeddings()
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    llm_logger = LLMLogger()
    Settings.callback_manager = CallbackManager([llama_debug, llm_logger])
    return Settings
