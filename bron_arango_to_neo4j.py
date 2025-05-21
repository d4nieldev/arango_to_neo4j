import re
from typing import Optional, List

from arango_to_neo4j import ArangoToNeo4j, get_neo4j_driver
import bron_node_transformers
import bron_edge_transformers
import bron_node_summaries

import constants as c


def get_summary_of_node(node_data: dict, node_type: str) -> str:
    node_summary_func = getattr(bron_node_summaries, node_type + "_summary")
    node_summary = node_summary_func(node_data=node_data)
    return node_summary


class BronArangoToNeo4j(ArangoToNeo4j):
    def __init__(self):
        super().__init__(graph_name='bron')
        self.__edge_type_to_rel_type = {
            'TacticTechnique': 'IS_ACHIEVED_BY_TECHNIQUE',
            'GroupSoftware': 'USED_SOFTWARE',
            'GroupTechnique': 'USED_TECHNIQUE',
            'SoftwareTechnique': 'IMPLEMENTS_TECHNIQUE',
            'TechniqueTechnique': 'IS_REFINED_BY_SUB_TECHNIQUE',
            'D3fend_mitigationTechnique': 'DEFENDS_AGAINST_TECHNIQUE',
            'Engage_goalEngage_approach': 'IS_ACHIEVED_BY_APPROACH',
            'Engage_approachEngage_activity': 'IS_IMPLEMENTED_BY_ACTIVITY',
            'TechniqueEngage_activity': 'IS_ADDRESSED_BY_ACTIVITY',
            'TechniqueTechnique_mitigation': 'TECHNIQUE_IS_MITIGATED_BY',
            'TechniqueTechnique_detection': 'TECHNIQUE_IS_DETECTED_BY',
            'TechniqueCapec': 'IS_REPRESENTED_AS_ATTACK_PATTERN',
            'CapecCapec': 'IS_PARENT_OF_ATTACK_PATTERN',
            'CapecCapec_mitigation': 'ATTACK_PATTERN_IS_MITIGATED_BY',
            'CapecCapec_detection': 'ATTACK_PATTERN_IS_DETECTED_BY',
            'CapecCwe': 'EXPLOITS_WEAKNESS',
            'CweCwe': 'IS_PARENT_OF_WEAKNESS',
            'CweCwe_mitigation': 'WEAKNESS_IS_MITIGATED_BY',
            'CweCwe_detection': 'WEAKNESS_IS_DETECTED_BY',
            'CweCve': 'IS_BEING_EXPLOITED_IN_VULNERABILITY',
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
        relation, src_node_types, dst_node_types = self._arango_edge_type_info(
            edge_type=edge_type)
        return transformed_edge, relation, src_node_types[0], dst_node_types[0]

    def _arango_edge_type_info(self, edge_type: str) -> tuple[str, list[str], list[str]]:
        rel_type = self.__edge_type_to_rel_type[edge_type]
        src_layer, dst_layer = re.sub(
            r'(?<!^)(?=[A-Z])', ' ', edge_type).split()
        return rel_type, [src_layer.lower()], [dst_layer.lower()]

    def build_neo4j(self, ignore_files: Optional[List[str]] = None) -> None:
        super().build_neo4j(ignore_files=ignore_files)
        with get_neo4j_driver() as driver:
            driver.verify_connectivity()

            driver.execute_query(
                "MATCH (c:capec)-[]->(n:capec_mitigation) SET n.capec_id = c.original_id")
            driver.execute_query(
                "MATCH (c:capec)-[]->(n:capec_detection) SET n.capec_id = c.original_id")

            driver.execute_query(
                "MATCH (c:cwe)-[]->(n:cwe_mitigation) SET n.cwe_id = c.original_id")
            driver.execute_query(
                "MATCH (c:cwe)-[]->(n:cwe_detection) SET n.cwe_id = c.original_id")


if __name__ == '__main__':
    generator = BronArangoToNeo4j()
    generator.load_graph(from_file=True, rewrite_file=False)
    # generator.generate_instructions(load_if_exists=True)
    # generator.build_neo4j()
    generator.validate_build_successful()
    from bron_edge_transformers import EdgeData
    edge_data = EdgeData()
    print({t: edge_data.found[t] / edge_data.total[t] for t in edge_data.found})
    pass
    # generator.add_embeddings()
