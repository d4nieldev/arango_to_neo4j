from arango_to_neo4j import ArangoToNeo4j


class BronArangoToNeo4j(ArangoToNeo4j):
    def __init__(self):
        super().__init__(graph_name='bron',
                         ignored_node_types=('engage_activity', 'engage_approach', 'engage_goal'))

    def _arango_node_transformer(self, arango_node: dict) -> dict:
        return arango_node

    def _arango_edge_transformer(self, arango_edge: dict) -> dict:
        return arango_edge


if __name__ == '__main__':
    generator = BronArangoToNeo4j()
