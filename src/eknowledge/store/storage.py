from llama_index.core import StorageContext

from eknowledge.store.doc_store import get_doc_store
from eknowledge.store.graph_store import get_graph_store
from eknowledge.store.vector_store import get_vector_store


def get_storage_context():
    doc_store = get_doc_store()
    vector_store = get_vector_store()
    graph_store = get_graph_store()
    storage_context = StorageContext.from_defaults(
        docstore=doc_store, vector_store=vector_store, property_graph_store=graph_store
    )
    return storage_context
