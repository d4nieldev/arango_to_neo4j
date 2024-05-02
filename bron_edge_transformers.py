"""
The methods in this file are transformers for the edges in the arango BRON graph.
Each transformer function returns the transformed edge, and the relation (text on the edge).
It is important that the name of the function will be {edge_datatype}_trans.
The methods should NOT override the properties _id, _from, _to !!
"""


def TacticTechnique_trans(edge: dict) -> (dict, str):
    pass


def GroupSoftware_trans(edge: dict) -> (dict, str):
    pass


def GroupTechnique_trans(edge: dict) -> (dict, str):
    pass


def SoftwareTechnique_trans(edge: dict) -> (dict, str):
    pass


def TechniqueTechnique_trans(edge: dict) -> (dict, str):
    pass


def TechniqueD3fend_mitigation_trans(edge: dict) -> (dict, str):
    pass


def TechniqueTechnique_mitigation_trans(edge: dict) -> (dict, str):
    pass


def TechniqueTechnique_detection_trans(edge: dict) -> (dict, str):
    pass


def TechniqueCapec_trans(edge: dict) -> (dict, str):
    pass


def CapecCapec_trans(edge: dict) -> (dict, str):
    pass


def CapecCapec_mitigation_trans(edge: dict) -> (dict, str):
    pass


def CapecCapec_detection_trans(edge: dict) -> (dict, str):
    pass


def CapecCwe_detection_trans(edge: dict) -> (dict, str):
    pass


def CweCwe_trans(edge: dict) -> (dict, str):
    pass


def CweCwe_mitigation_trans(edge: dict) -> (dict, str):
    pass


def CweCwe_detection_trans(edge: dict) -> (dict, str):
    pass


def CweCve_trans(edge: dict) -> (dict, str):
    pass


def CveCpe_trans(edge: dict) -> (dict, str):
    pass
