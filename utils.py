import os
import json
import logging as log

log.getLogger().setLevel(log.INFO)


def read_file(file_path: str, is_json: bool = False, split_by_new_line: bool = False) -> (str or dict or list):
    try:
        with open(file_path, "r") as f:
            if is_json:
                data = json.load(f)
            else:
                data = f.read()
                if split_by_new_line:
                    data = data.split("\n")
        return data
    except Exception as e:
        log.error(f"Error: file: {file_path} {repr(e)}")
        raise e


def write_file(file_path: str, data: object, is_json: bool = False, create_path_if_not_exist: bool = True) -> None:
    try:
        if create_path_if_not_exist:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            if is_json:
                json.dump(data, f, indent=4)
            else:
                if isinstance(data, list):
                    f.write('\n'.join(data))
                else:
                    f.write(data)
    except Exception as e:
        log.error(f"Error: {repr(e)}")
        raise e


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


