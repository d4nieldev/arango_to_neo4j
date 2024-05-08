"""
The methods in this file are transformers for the edges in the arango BRON graph.
Each transformer function returns the transformed edge, and the relation.
It is important that the name of the function will be {edge_datatype}_trans.
The methods should NOT override the properties _id, _from, _to !!
"""


def TacticTechnique_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_ACHIEVED_BY'


def GroupSoftware_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_USING_SOFTWARE'


def GroupTechnique_trans(edge: dict) -> tuple[dict, str]:
    # TODO: add description on this edge type
    return {}, 'USED_TECHNIQUE'


def SoftwareTechnique_trans(edge: dict) -> tuple[dict, str]:
    return {'description': edge['description']}, 'IS_USING_TECHNIQUE'


def TechniqueTechnique_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_PARENT_OF_SUB_TECHNIQUE'


def TechniqueD3fend_mitigation_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_MITIGATED_BY_D3FEND_MITIGATION'


def TechniqueTechnique_mitigation_trans(edge: dict) -> tuple[dict, str]:
    return {'description': edge['metadata']['description']}, 'IS_MITIGATED_BY'


def TechniqueTechnique_detection_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_DETECTED_BY'


def TechniqueCapec_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_USED_BY_ATTACK_PATTERN'


def CapecCapec_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_PARENT_OF_ATTACK_PATTERN'


def CapecCapec_mitigation_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_MITIGATED_BY'


def CapecCapec_detection_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_DETECTED_BY'


def CapecCwe_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'EXPLOITS_WEAKNESS'


def CweCwe_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_PARENT_OF_WEAKNESS'


def CweCwe_mitigation_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_MITIGATED_BY'


def CweCwe_detection_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_DETECTED_BY'


def CweCve_trans(edge: dict) -> tuple[dict, str]:
    # TODO: add description on this edge type
    return {}, 'BEING_EXPLOITED_IN'


def CveCpe_trans(edge: dict) -> tuple[dict, str]:
    return {}, 'IS_COMPROMISING_PLATFORM'
