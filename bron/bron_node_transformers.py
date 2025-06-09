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
        'description': pick_longest_description(node['metadata']['description'], node['metadata']['short_description']),
    }


def technique_mitigation_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
        'description': node['metadata']['description'],
    }


def technique_detection_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
        'description': node['metadata']['description'],
    }


def d3fend_mitigation_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
        'description': node['metadata']['description'],
    }


def capec_trans(node: dict) -> dict:
    return {
        'original_id': f"CAPEC-{node['original_id']}",
        'name': node['name'],
        'consequences': node['metadata']['consequences'],
        'description': pick_longest_description(node['metadata']['description'], node['metadata']['extended_description']),
        'likelihood_of_attack': node['metadata']['likelihood_of_attack'],
        'resources_required': node['metadata']['resources_required'],
        'skills_required': node['metadata']['skills_required'],
        'typical_severity': node['metadata']['typical_severity'],
    }


def capec_mitigation_trans(node: dict) -> dict:
    return {
        'name': node['name'],
        'description': node['metadata'],
    }


def capec_detection_trans(node: dict) -> dict:
    return {
        'name': node['original_id'],
        'description': node['metadata'],
    }


def cwe_trans(node: dict) -> dict:
    return {
        'original_id': f"CWE-{node['original_id']}",
        'name': node['name'],
        'description': pick_longest_description(node['metadata']['short_description'], node['metadata']['description']),
        'likelihood_of_exploit': node['metadata']['likeliehood_of_exploit'],
        'applicable_platforms': node['metadata']['applicable_platform'],
        'common_consequences': node['metadata']['common_consequences'],
    }


def cwe_mitigation_trans(node: dict) -> dict:
    return {
        'name': node['name'],
        'phase': node['metadata']['Phase'],
        'description': node['metadata']['Description'],
    }


def cwe_detection_trans(node: dict) -> dict:
    return {
        'name': node['name'],
        'method': node['metadata']['Method'],
        'description': node['metadata']['Description'],
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


def engage_activity_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
    }


def engage_approach_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
    }


def engage_goal_trans(node: dict) -> dict:
    return {
        'original_id': node['original_id'],
        'name': node['name'],
    }
