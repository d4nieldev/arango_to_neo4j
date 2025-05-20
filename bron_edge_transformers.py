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
        self.group_technique_df = stixToDf.groupsToDf(attackdata)['techniques used']
        self.software_technique_df = stixToDf.softwareToDf(attackdata)['techniques used']
        self.mitigation_technique_df = stixToDf.mitigationsToDf(attackdata)['techniques addressed']
        self.cwe_cve_dict = {}


def TacticTechnique_trans(edge: dict) -> dict:
    return {}


def GroupSoftware_trans(edge: dict) -> dict:
    return {}


def GroupTechnique_trans(edge: dict) -> dict:
    src_id = edge['_from'].split('/')[-1]
    dst_id = edge['_to'].split('/')[-1]
    group_technique_df = EdgeData().group_technique_df
    
    group_technique = group_technique_df[
        group_technique_df['source_id'] == src_id & 
        group_technique_df['target_id'] == dst_id
    ]
    if not group_technique.empty:
        return {'description': group_technique.iloc[0]['mapping description']}
    log.warning(f"Link between group {src_id} and technique {dst_id} not found in MITRE ATT&CK")
    return {}


def SoftwareTechnique_trans(edge: dict) -> dict:
    src_id = edge['_from'].split('/')[-1]
    dst_id = edge['_to'].split('/')[-1]
    software_technique_df = EdgeData().software_technique_df

    software_technique = software_technique_df[
        software_technique_df['source_id'] == src_id &
        software_technique_df['target_id'] == dst_id
    ]
    if not software_technique.empty:
        return {'description': software_technique.iloc[0]['mapping description']}
    log.warning(f"Link between software {src_id} and technique {dst_id} not found in MITRE ATT&CK")
    return {}


def TechniqueTechnique_trans(edge: dict) -> dict:
    return {}


def D3fend_mitigationTechnique_trans(edge: dict) -> dict:
    return {}


def Engage_goalEngage_approach(edge: dict) -> dict:
    return {}


def Engage_approachEngage_activity(edge: dict) -> dict:
    return {}


def TechniqueEngage_activity(edge: dict) -> dict:   
    return {}


def TechniqueTechnique_mitigation_trans(edge: dict) -> dict:
    technique_id = edge['_from'].split('/')[-1]
    mitigation_id = edge['_to'].split('/')[-1]
    mitigation_technique_df = EdgeData().mitigation_technique_df

    mitigation_technique = mitigation_technique_df[
        mitigation_technique_df['source_id'] == mitigation_id &
        mitigation_technique_df['target_id'] == technique_id
    ]
    if not mitigation_technique.empty:
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
    cwe_cve_dict = EdgeData().cwe_cve_dict

    if cwe_id in cwe_cve_dict and cwe_cve_dict[cwe_id] and cve_id in cwe_cve_dict[cwe_id]:
        return {'description': cwe_cve_dict[cwe_id][cve_id]}
    elif cwe_id in cwe_cve_dict and cwe_cve_dict[cwe_id] and cve_id not in cwe_cve_dict[cwe_id]:
        log.warning(f"Connection between CWE {cwe_id} and CVE {cve_id} not found in CWE API")
    elif cwe_id not in cwe_cve_dict:
        cwe_cve_dict[cwe_id] = {}
        resp = requests.get(f"https://cwe-api.mitre.org/api/v1/cwe/weakness/{cwe_id.split('-')[1]}")
        if resp.status_code == 404:
            log.warning(f"Could not fine CWE {cwe_id} in CWE API")
        else:
            # successfully fetched CWE data
            related_cves = resp.json()['Weaknesses'][0].get('ObservedExamples', [])
            for cve_ref in related_cves:
                cwe_cve_dict[cwe_id][cve_ref['Reference']] = cve_ref['Description']
    return {}


def CveCpe_trans(edge: dict) -> dict:
    return {}
