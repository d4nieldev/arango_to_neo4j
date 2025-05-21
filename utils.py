import os
import json
from pathlib import Path
import logging as log
from typing import Union, List, Dict

log.getLogger().setLevel(log.INFO)
log.getLogger('httpx').setLevel(log.WARNING)


def read_file(file_path: Path, is_json: bool = False, split_by_new_line: bool = False) -> Union[Dict, List, str]:
    if is_json:
        data = json.loads(file_path.read_text())
    else:
        data = file_path.read_text()
        if split_by_new_line:
            data = data.split("\n")
    return data


def write_file(file_path: Path, data: object, is_json: bool = False, create_path_if_not_exist: bool = True) -> None:
    if create_path_if_not_exist:
        file_path.parent.mkdir()
    
    if is_json:
        file_path.write_text(json.dumps(data, indent=2))
    else:
        if isinstance(data, list):
            file_path.write_text('\n'.join(data))
        else:
            file_path.write_text(repr(data))


def pick_longest_description(desc1: str, desc2: str):
    if desc1 is None:
        return desc2.strip()
    if desc2 is None:
        return desc1.strip()

    stripped_desc1 = desc1.strip()
    stripped_desc2 = desc2.strip()
    return stripped_desc1 if len(stripped_desc1) > len(stripped_desc2) else stripped_desc2


def create_markdown_table(items: list[dict[str, str]]) -> str:
    assert len(items) > 0, "You shouldn't make tables with no data..."

    keys = set()
    for item in items:
        keys.update(set(item.keys()))

    keys = list(keys)

    # table header
    table = "|"
    for key in keys:
        table += f" {key.capitalize()} |"

    table += '\n|'
    for key in keys:
        table += f" {'-'*len(key)} |"

    # table body
    for item in items:
        table += '\n|'
        for key in keys:
            table += f" {item.get(key, "")} |"

    return table

