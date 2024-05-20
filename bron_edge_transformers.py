"""
The methods in this file are transformers for the edges in the arango BRON graph.
Each transformer function returns the transformed edge.
It is important that the name of the function will be {edge_datatype}_trans.
The methods should NOT override the properties _id, _from, _to !!
"""


def TacticTechnique_trans(edge: dict) -> dict:
    return {}


def GroupSoftware_trans(edge: dict) -> dict:
    # TODO: maybe use techniques used to generate an explanation on this edge type
    return {}


def GroupTechnique_trans(edge: dict) -> dict:
    # TODO: add description on this edge type
    return {}


def SoftwareTechnique_trans(edge: dict) -> dict:
    return {'description': edge['description']}


def TechniqueTechnique_trans(edge: dict) -> dict:
    return {}


def TechniqueD3fend_mitigation_trans(edge: dict) -> dict:
    return {}


def TechniqueTechnique_mitigation_trans(edge: dict) -> dict:
    return {'description': edge['metadata']['description']}


def TechniqueTechnique_detection_trans(edge: dict) -> dict:
    return {}


def TechniqueCapec_trans(edge: dict) -> dict:
    return {}


def CapecCapec_trans(edge: dict) -> dict:
    return {}


def CapecCapec_mitigation_trans(edge: dict) -> dict:
    return {}


def CapecCapec_detection_trans(edge: dict) -> dict:
    return {}


def CapecCwe_trans(edge: dict) -> dict:
    return {}


def CweCwe_trans(edge: dict) -> dict:
    return {}


def CweCwe_mitigation_trans(edge: dict) -> dict:
    return {}


def CweCwe_detection_trans(edge: dict) -> dict:
    return {}


def CweCve_trans(edge: dict) -> dict:
    # TODO: add description on this edge type
    return {}


def CveCpe_trans(edge: dict) -> dict:
    # TODO: add description on this edge type
    return {}
