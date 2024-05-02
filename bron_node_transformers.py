"""
The methods in this file are transformers for the nodes in the arango BRON graph.
It is important that the name of the function will be {node_datatype}_trans.
The methods should NOT override the property _id !!
"""


def tactic_trans(node: dict) -> dict:
    pass


def group_trans(node: dict) -> dict:
    pass


def software_trans(node: dict) -> dict:
    pass


def technique_trans(node: dict) -> dict:
    pass


def technique_mitigation_trans(node: dict) -> dict:
    pass


def technique_detection_trans(node: dict) -> dict:
    pass


def d3fend_mitigation_trans(node: dict) -> dict:
    pass


def capec_trans(node: dict) -> dict:
    pass


def capec_mitigation_trans(node: dict) -> dict:
    pass


def capec_detection_trans(node: dict) -> dict:
    pass


def cwe_trans(node: dict) -> dict:
    pass


def cwe_mitigation_trans(node: dict) -> dict:
    pass


def cwe_detection_trans(node: dict) -> dict:
    pass


def cve_trans(node: dict) -> dict:
    pass


def cpe_trans(node: dict) -> dict:
    pass
