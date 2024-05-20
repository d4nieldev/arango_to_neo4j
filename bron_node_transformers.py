"""
The methods in this file are transformers for the nodes in the arango BRON graph.
It is important that the name of the function will be {node_datatype}_trans.
To get the full node, you should add the _id field!
"""
from utils import pick_longest_description


def tactic_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
        'description': pick_longest_description(node['metadata']['short_description'], node['metadata']['description'])
    }


def group_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
        'description': node['metadata']['description'],
        'aliases': node['metadata']['aliases']
    }


def software_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
        'description': node['metadata']['description'],
        'software_type': node['metadata']['type']
    }


def technique_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
        'description': node['metadata']['description'],
    }


def technique_mitigation_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
        'description': node['metadata']['description'],
    }


def technique_detection_trans(node: dict) -> dict:
    return {
        'technique_id': node['technique_id'],
        'description': node['detections'],
    }


def d3fend_mitigation_trans(node: dict) -> dict:
    # TODO: add how it works
    return {
        'original_id': node['original_id'],
        'name': node['name'],
        'description': node['metadata']['definition'],
        'synonyms': list(node['metadata']['synonyms'])
    }


def capec_trans(node: dict) -> dict:
    return {
        'original_id': f"CAPEC-{node['original_id']}",
        'name': node['name'],
        'description': pick_longest_description(node['metadata']['short_description'], node['metadata']['description']),
        'consequences': node['metadata']['consequences'],
        'likelihood_of_attack': node['metadata']['likelihood_of_attack'],
        'resources_required': node['metadata']['resources_required'],
        'skills_required': node['metadata']['skills_required'],
        'typical_severity': node['metadata']['typical_severity'],
    }


def capec_mitigation_trans(node: dict) -> dict:
    return {
        'capec_id': node['original_id'],
        'description': node['metadata'],
    }


def capec_detection_trans(node: dict) -> dict:
    return {
        'capec_id': node['original_id'],
        'description': node['metadata'],
    }


def cwe_trans(node: dict) -> dict:
    return {
        'original_id': f"CWE-{node['original_id']}",
        'name': node['name'],
        'description': pick_longest_description(node['metadata']['short_description'], node['metadata']['description']),
        'likelihood_of_exploit': node['metadata']['likeliehood_of_exploit'],
        'applicable_platforms': node['metadata']['applicable_platform'], # TODO: fix this field
        'common_consequences': node['metadata']['common_consequences'],
    }


def cwe_mitigation_trans(node: dict) -> dict:
    return {
        'cwe_id': node['original_id'],
        'description': node['metadata']['Description'],
        'phase': node['metadata']['Phase'],
    }


def cwe_detection_trans(node: dict) -> dict:
    return {
        'cwe_id': node['original_id'],
        'description': node['metadata']['Description'],
        'method': node['metadata']['Method'],
    }


def cve_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'description': node['metadata']['description'],
        'severity': node['metadata']['weight'],
    }


def cpe_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'product': node['metadata']['product'],
        'vendor': node['metadata']['vendor'],
        'version': node['metadata']['version'],
    }

# TODO: add datasources, and descriptions on TechniqueTechnique_mitigation
