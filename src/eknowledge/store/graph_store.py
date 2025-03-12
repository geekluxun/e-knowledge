from llama_index.graph_stores.neo4j import Neo4jPGStore


def get_graph_store():
    graph_store = Neo4jPGStore(
        username="neo4j",
        password="12345678",
        url="bolt://localhost:7689",
    )
    return graph_store


if __name__ == "__main__":
    pass
