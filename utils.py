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


def get_src_and_dst_types_from_edge_type(edge_type: str) -> tuple[str, str]:
    src_type = edge_type[0]

    i = 1
    while i < len(edge_type) and edge_type[i].islower():
        src_type += edge_type[i]
        i += 1

    assert i < len(edge_type), f'invalid arango edge type name: {edge_type}'
    dst_type = edge_type[i:]

    return src_type.lower(), dst_type.lower()


def pick_longest_description(desc1: str, desc2: str):
    if desc1 is None:
        return desc2.strip()
    if desc2 is None:
        return desc1.strip()

    stripped_desc1 = desc1.strip()
    stripped_desc2 = desc2.strip()
    return stripped_desc1 if len(stripped_desc1) > len(stripped_desc2) else stripped_desc2



