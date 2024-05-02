from arango_to_neo4j import ArangoToNeo4j
import bron_node_transformers
import bron_edge_transformers

import constants as c


class BronArangoToNeo4j(ArangoToNeo4j):
    def __init__(self):
        super().__init__(graph_name='bron',
                         ignored_node_types=('engage_activity', 'engage_approach', 'engage_goal'))

    def _arango_node_transformer(self, arango_node: dict, node_type: str) -> dict:
        transformer = getattr(bron_node_transformers, node_type + "_trans")
        return transformer(node=arango_node)

    def _arango_edge_transformer(self, arango_edge: dict, edge_type: str) -> tuple[dict, str]:
        transformer = getattr(bron_edge_transformers, edge_type + "_trans")
        return transformer(edge=arango_edge)


if __name__ == '__main__':
    generator = BronArangoToNeo4j()
    generator.load_graph(from_file=True, rewrite_file=False)
