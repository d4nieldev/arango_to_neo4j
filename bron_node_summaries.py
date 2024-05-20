"""
The methods in this file provide markdown summaries for the nodes in the neo4j BRON graph.
It is important that the name of the function will be {node_datatype}_summary.
"""
import json
from utils import create_markdown_table


def group_summary(node_data: dict) -> str:
    output = f"# MITRE ATT&CK Group \"{node_data['name']}\" ({node_data['original_id']})\n"
    output += f"## Description\n{node_data['description']}"
    if len(node_data['aliases']) > 0:
        output += f"## Aliases\n{'\n'.join([f'* {a}' for a in node_data['aliases']])}\n"

    return output


def software_summary(node_data: dict) -> str:
    output = f"# MITRE ATT&CK Software \"{node_data['name']}\" ({node_data['original_id']})\n"
    output += f"## Description\n{node_data['description']}"
    output += f"## Type\n{node_data['software_type']}\n"

    return output


def tactic_summary(node_data: dict) -> str:
    output = f"# MITRE ATT&CK Tactic \"{node_data['name']}\" ({node_data['original_id']})\n"
    output += f"## Description\n{node_data['description']}"

    return output


def technique_summary(node_data: dict) -> str:
    technique_str = 'Sub-Technique' if '.' in node_data['original_id'] else 'Technique'
    output = f"# MITRE ATT&CK {technique_str} \"{node_data['name']}\" ({node_data['original_id']})\n"
    output += f"## Description\n{node_data['description']}"

    return output


def d3fend_mitigation_summary(node_data: dict) -> str:
    output = f"# MITRE D3FEND Mitigation \"{node_data['name']}\" ({node_data['original_id']})\n"
    output += f"## Description\n{node_data['description']}\n"
    output += f"## Synonyms\n{'\n'.join([f'* {s}' for s in node_data['synonyms']])}"

    return output


def technique_mitigation_summary(node_data: dict) -> str:
    output = f"# MITRE ATT&CK Technique Mitigation \"{node_data['name']}\" ({node_data['original_id']})\n"
    output += f"## Description\n{node_data['description']}"

    return output


def technique_detection_summary(node_data: dict) -> str:
    technique_str = 'sub-technique' if '.' in node_data['original_id'] else 'technique'
    output = f"# Detections for {technique_str} {node_data['technique_id']}\n"
    output += node_data['description']

    return output


def capec_summary(node_data: dict) -> str:
    output = f"# Attack Pattern \"{node_data['name']}\" ({node_data['original_id']})\n"
    output += f"## Description\n{node_data['description']}\n"
    if node_data['likelihood_of_attack'].strip() != "":
        output += f"## Likelihood of Attack\n{node_data['likelihood_of_attack']}\n"
    if node_data['typical_severity'].strip() != "":
        output += f"## Typical Severity\n{node_data['typical_severity']}\n"

    skills_required: list[dict] = [json.loads(skill) for skill in node_data['skills_required']]
    if len(skills_required) > 0:
        skills_table = create_markdown_table(items=skills_required)
        output += f"## Skills Required\n{skills_table}\n"

    resources_required = node_data['resources_required']
    if (len(resources_required) > 0 and
            resources_required[0] != "None: No specialized resources are required to execute this type of attack."):
        output += f"## Resources Required\n{'\n'.join([f'* {r}' for r in resources_required])}\n"

    consequences: list[dict] = [json.loads(con) for con in node_data['consequences']]
    if len(consequences) > 0:
        consequences_table = create_markdown_table(items=consequences)
        output += f"## Consequences\n{consequences_table}"

    return output


def capec_mitigation_summary(node_data: dict) -> str:
    output = f"# A way to mitigate the attack pattern {node_data['capec_id']}\n"
    output += node_data['description']

    return output


def capec_detection_summary(node_data: dict) -> str:
    output = f"# A way to detect the attack pattern {node_data['capec_id']}\n"
    output += node_data['description']

    return output


def cwe_summary(node_data: dict) -> str:
    output = f"# Weakness \"{node_data['name']}\" ({node_data['original_id']})\n"
    output += f"## Description\n{node_data['description']}\n"

    common_consequences = [json.loads(con) for con in node_data['common_consequences']]
    if len(common_consequences) > 0:
        common_consequences_table = create_markdown_table(items=common_consequences)
        output += f"## Common Consequences\n{common_consequences_table}\n"

    if node_data['likelihood_of_exploit	'].strip() != "":
        output += f"## Likelihood of Exploit\n{node_data['likelihood_of_exploit']}\n"

    return output


def cwe_mitigation_summary(node_data: dict) -> str:
    output = f"# A way to mitigate the weakness {node_data['cwe_id']}\n"
    output += f"## Phase: {node_data['phase']}\n"
    output += node_data['description']

    return output


def cwe_detection_summary(node_data: dict) -> str:
    output = f"# A way to detect the attack pattern {node_data['capec_id']}\n"
    output += node_data['description']

    return output


def cve_summary(node_data: dict) -> str:
    output = f"# Vulnerability {node_data['original_id']}\n"
    output += f"## Description\n{node_data['description']}"
    output += f"## Severity\n{node_data['severity']}/10"

    return output


def cpe_summary(node_data: dict) -> str:
    part = node_data['original_id'].split(':')[2]
    product_type = "Product"
    if part == 'a':
        product_type = "Application"
    elif part == 'h':
        product_type = "Hardware"
    elif part == 'o':
        product_type = "Operating system"

    output = f"# Platform {node_data['original_id']}"
    output += f"{product_type} {node_data['product']} by {node_data['vendor']} version {node_data['version']}"

    return output
