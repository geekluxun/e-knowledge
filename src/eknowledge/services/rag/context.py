from llama_index.core import StorageContext

from eknowledge.services.rag.store.doc_store import get_doc_store
from eknowledge.services.rag.store.graph_store import get_graph_store
from eknowledge.services.rag.store.vector_store import get_vector_store


class _MyStorageContext:
    _storage_context = None

    def storageContext(self):
        if self._storage_context is None:
            doc_store = get_doc_store()
            vector_store = get_vector_store()
            graph_store = get_graph_store()
            self._storage_context = StorageContext.from_defaults(
                docstore=doc_store, vector_store=vector_store, property_graph_store=graph_store
            )
        return self._storage_context


# 单例
MyStorageContext = _MyStorageContext()
