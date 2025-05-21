"""
The methods in this file are transformers for the edges in the arango BRON graph.
Each transformer function returns the transformed edge.
It is important that the name of the function will be {edge_datatype}_trans.
The methods should NOT override the properties _id, _from, _to !!
"""

from mitreattack.attackToExcel import attackToExcel
import mitreattack.attackToExcel.stixToDf as stixToDf
import requests
import logging as log
from singleton import SingletonMeta


class EdgeData(metaclass=SingletonMeta):
    def __init__(self):
        attackdata = attackToExcel.get_stix_data('enterprise-attack')
        self.group_technique_df = stixToDf.groupsToDf(attackdata)[
            'techniques used']
        self.software_technique_df = stixToDf.softwareToDf(attackdata)[
            'techniques used']
        self.mitigation_technique_df = stixToDf.mitigationsToDf(attackdata)[
            'techniques addressed']
        self.cwe_cve_dict = {}

        self.total = {}
        self.found = {}


def TacticTechnique_trans(edge: dict) -> dict:
    return {}


def GroupSoftware_trans(edge: dict) -> dict:
    return {}


def GroupTechnique_trans(edge: dict) -> dict:
    src_id = edge['_from'].split('/')[-1]
    dst_id = edge['_to'].split('/')[-1]

    edge_data = EdgeData()
    group_technique_df = edge_data.group_technique_df
    edge_data.total['group_technique'] = edge_data.total.get('group_technique', 0) + 1

    group_technique = group_technique_df[
        (group_technique_df['source ID'] == src_id) &
        (group_technique_df['target ID'] == dst_id)
    ]
    if not group_technique.empty:
        edge_data.found['group_technique'] = edge_data.found.get('group_technique', 0) + 1
        return {'description': group_technique.iloc[0]['mapping description']}
    log.warning(f"Link between group {src_id} and technique {dst_id} not found in MITRE ATT&CK")
    return {}


def SoftwareTechnique_trans(edge: dict) -> dict:
    src_id = edge['_from'].split('/')[-1]
    dst_id = edge['_to'].split('/')[-1]

    edge_data = EdgeData()
    software_technique_df = edge_data.software_technique_df
    edge_data.total['software_technique'] = edge_data.total.get('software_technique', 0) + 1

    software_technique = software_technique_df[
        (software_technique_df['source ID'] == src_id) &
        (software_technique_df['target ID'] == dst_id)
    ]
    if not software_technique.empty:
        edge_data.found['software_technique'] = edge_data.found.get('software_technique', 0) + 1
        return {'description': software_technique.iloc[0]['mapping description']}
    log.warning(f"Link between software {src_id} and technique {dst_id} not found in MITRE ATT&CK")
    return {}


def TechniqueTechnique_trans(edge: dict) -> dict:
    return {}


def D3fend_mitigationTechnique_trans(edge: dict) -> dict:
    return {}


def Engage_goalEngage_approach_trans(edge: dict) -> dict:
    return {}


def Engage_approachEngage_activity_trans(edge: dict) -> dict:
    return {}


def TechniqueEngage_activity_trans(edge: dict) -> dict:
    return {}


def TechniqueTechnique_mitigation_trans(edge: dict) -> dict:
    technique_id = edge['_from'].split('/')[-1]
    mitigation_id = edge['_to'].split('/')[-1]

    edge_data = EdgeData()
    mitigation_technique_df = edge_data.mitigation_technique_df
    edge_data.total['mitigation_technique'] = edge_data.total.get('mitigation_technique', 0) + 1

    mitigation_technique = mitigation_technique_df[
        (mitigation_technique_df['source ID'] == mitigation_id) &
        (mitigation_technique_df['target ID'] == technique_id)
    ]
    if not mitigation_technique.empty:
        edge_data.found['mitigation_technique'] = edge_data.found.get('mitigation_technique', 0) + 1
        return {'description': mitigation_technique.iloc[0]['mapping description']}
    log.warning(f"Link between mitigation {mitigation_id} and technique {technique_id} not found in MITRE ATT&CK")
    return {}


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
    cwe_id = edge['_from'].split('/')[-1]
    cve_id = edge['_to'].split('/')[-1]

    edge_data = EdgeData()
    cwe_cve_dict = edge_data.cwe_cve_dict
    edge_data.total['cwe_cve'] = edge_data.total.get('cwe_cve', 0) + 1

    if cwe_id not in cwe_cve_dict:
        cwe_cve_dict[cwe_id] = {}
        resp = requests.get(f"https://cwe-api.mitre.org/api/v1/cwe/weakness/{cwe_id.split('-')[1]}")
        if resp.status_code == 404:
            log.warning(f"Could not fine CWE {cwe_id} in CWE API")
        else:
            # successfully fetched CWE data
            weakness_json = resp.json()['Weaknesses'][0]
            if weakness_json['MappingNotes']['Usage'] != 'Allowed':
                log.warning(f"Mapping notes for CWE {cwe_id} is {weakness_json['MappingNotes']['Usage'].lower()}")
            related_cves = weakness_json.get('ObservedExamples', [])
            for cve_ref in related_cves:
                cwe_cve_dict[cwe_id][cve_ref['Reference']] = cve_ref['Description']

    if cwe_cve_dict[cwe_id] and cve_id in cwe_cve_dict[cwe_id]:
        edge_data.found['cwe_cve'] = edge_data.found.get('cwe_cve', 0) + 1
        return {'description': cwe_cve_dict[cwe_id][cve_id]}
    elif cwe_cve_dict[cwe_id] and cve_id not in cwe_cve_dict[cwe_id]:
        log.debug(
            f"Connection between CWE {cwe_id} and CVE {cve_id} not found in CWE API")

    return {}


def CveCpe_trans(edge: dict) -> dict:
    return {}
