import re

from arango_to_neo4j import ArangoToNeo4j, get_neo4j_driver
import bron_node_transformers
import bron_edge_transformers
import bron_node_summaries

import constants as c


class BronArangoToNeo4j(ArangoToNeo4j):
    def __init__(self):
        super().__init__(graph_name='bron',
                         ignored_node_types=('engage_activity', 'engage_approach', 'engage_goal'))
        self.__edge_type_to_rel_type = {
            'TacticTechnique': 'IS_ACHIEVED_BY',
            'GroupSoftware': 'IS_USING_SOFTWARE',
            'GroupTechnique': 'USED_TECHNIQUE',
            'SoftwareTechnique': 'IS_USING_TECHNIQUE',
            'TechniqueTechnique': 'IS_PARENT_OF_SUB_TECHNIQUE',
            'TechniqueD3fend_mitigation': 'IS_MITIGATED_BY_D3FEND_MITIGATION',
            'TechniqueTechnique_mitigation': 'IS_MITIGATED_BY',
            'TechniqueTechnique_detection': 'IS_DETECTED_BY',
            'TechniqueCapec': 'IS_USED_BY_ATTACK_PATTERN',
            'CapecCapec': 'IS_PARENT_OF_ATTACK_PATTERN',
            'CapecCapec_mitigation': 'IS_MITIGATED_BY',
            'CapecCapec_detection': 'IS_DETECTED_BY',
            'CapecCwe': 'EXPLOITS_WEAKNESS',
            'CweCwe': 'IS_PARENT_OF_WEAKNESS',
            'CweCwe_mitigation': 'IS_MITIGATED_BY',
            'CweCwe_detection': 'IS_DETECTED_BY',
            'CweCve': 'BEING_EXPLOITED_IN',
            'CveCpe': 'IS_COMPROMISING_PLATFORM'
        }

    def _arango_node_transformer(self, arango_node: dict, node_type: str) -> dict:
        transformer = getattr(bron_node_transformers, node_type + "_trans")
        transformed_node = transformer(node=arango_node)
        transformed_node[c.ARANGO_NODE_ID_PROP] = arango_node[c.ARANGO_NODE_ID_PROP]
        return transformed_node

    def _arango_edge_transformer(self, arango_edge: dict, edge_type: str) -> tuple[dict, str, str, str]:
        transformer = getattr(bron_edge_transformers, edge_type + "_trans")
        transformed_edge = transformer(edge=arango_edge)
        transformed_edge[c.ARANGO_EDGE_ID_PROP] = arango_edge[c.ARANGO_EDGE_ID_PROP]
        transformed_edge[c.ARANGO_EDGE_FROM_ID_PROP] = arango_edge[c.ARANGO_EDGE_FROM_ID_PROP]
        transformed_edge[c.ARANGO_EDGE_TO_ID_PROP] = arango_edge[c.ARANGO_EDGE_TO_ID_PROP]
        relation, src_node_types, dst_node_types = self._arango_edge_type_info(edge_type=edge_type)
        return transformed_edge, relation, src_node_types[0], dst_node_types[0]

    def _arango_edge_type_info(self, edge_type: str) -> tuple[str, list[str], list[str]]:
        rel_type = self.__edge_type_to_rel_type[edge_type]
        src_layer, dst_layer = re.sub(r'(?<!^)(?=[A-Z])', ' ', edge_type).split()
        return rel_type, [src_layer.lower()], [dst_layer.lower()]

    def build_neo4j(self, ignore_files: list[str] = None) -> None:
        super().build_neo4j(ignore_files=ignore_files)
        with get_neo4j_driver() as driver:
            driver.verify_connectivity()

            driver.execute_query("MATCH (c:capec)-[]->(n:capec_mitigation) SET n.capec_id = c.original_id")
            driver.execute_query("MATCH (c:capec)-[]->(n:capec_detection) SET n.capec_id = c.original_id")

            driver.execute_query("MATCH (c:cwe)-[]->(n:cwe_mitigation) SET n.cwe_id = c.original_id")
            driver.execute_query("MATCH (c:cwe)-[]->(n:cwe_detection) SET n.cwe_id = c.original_id")

    def add_embeddings(self):
        regex_pattern = r'^(.+)_summary$'
        relevant_node_types = [re.match(regex_pattern, func_name).group(1)
                               for func_name in dir(bron_node_summaries) if re.match(regex_pattern, func_name)]
        relevant_node_types_str = ':' + ':'.join(relevant_node_types)
        with get_neo4j_driver() as driver:
            driver.verify_connectivity()
            # query to fetch one node from each type to see summary
            # query = ""
            # prev_nodes = []
            # for i, node_type in enumerate(relevant_node_types):
            #     node_name = f'{node_type}_node'
            #     prev_nodes.append(node_name)
            #     prev_node_types_str = ', '.join(prev_nodes)
            #
            #     query += f"MATCH ({node_name}:{node_type})\n"
            #     if i < len(relevant_node_types) - 1:
            #         query += f"WITH {prev_node_types_str}\n"
            #     else:
            #         query += f"RETURN {prev_node_types_str}\n"
            #     query += f"LIMIT 1\n"

            query = f"MATCH (n{relevant_node_types_str})"
            nodes, _, _ = driver.execute_query(query_=query)

            for node, in nodes:
                node_type = list(node.labels)[0]
                node_data = dict(node)
                node_summary_func = getattr(bron_node_summaries, node_type + "_summary")
                node_summary = node_summary_func(node_data=node_data)

                # TODO: calculate embeddings for the node summary text
                pass


if __name__ == '__main__':
    generator = BronArangoToNeo4j()
    # generator.load_graph(from_file=True, rewrite_file=False)
    # generator.generate_instructions()
    # generator.build_neo4j()
    # generator.validate_build_successful()
    generator.add_embeddings()
