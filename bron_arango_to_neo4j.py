from arango_to_neo4j import ArangoToNeo4j
import bron_node_transformers
import bron_edge_transformers

import constants as c
from utils import get_src_and_dst_types_from_edge_type


class BronArangoToNeo4j(ArangoToNeo4j):
    def __init__(self):
        super().__init__(graph_name='bron',
                         ignored_node_types=('engage_activity', 'engage_approach', 'engage_goal'))

    def _arango_node_transformer(self, arango_node: dict, node_type: str) -> dict:
        transformer = getattr(bron_node_transformers, node_type + "_trans")
        transformed_node = transformer(node=arango_node)
        transformed_node[c.ARANGO_NODE_ID_PROP] = arango_node[c.ARANGO_NODE_ID_PROP]
        return transformed_node

    def _arango_edge_transformer(self, arango_edge: dict, edge_type: str) -> tuple[dict, str, str, str]:
        transformer = getattr(bron_edge_transformers, edge_type + "_trans")
        transformed_edge, relation = transformer(edge=arango_edge)
        transformed_edge[c.ARANGO_EDGE_ID_PROP] = arango_edge[c.ARANGO_EDGE_ID_PROP]
        transformed_edge[c.ARANGO_EDGE_FROM_ID_PROP] = arango_edge[c.ARANGO_EDGE_FROM_ID_PROP]
        transformed_edge[c.ARANGO_EDGE_TO_ID_PROP] = arango_edge[c.ARANGO_EDGE_TO_ID_PROP]
        src_node_type, dst_node_type = get_src_and_dst_types_from_edge_type(edge_type=edge_type)
        return transformed_edge, relation, src_node_type, dst_node_type


if __name__ == '__main__':
    generator = BronArangoToNeo4j()
    # generator.load_graph(from_file=True, rewrite_file=False)
    # generator.generate_instructions()
    generator.build_neo4j()
