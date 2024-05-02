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
                f.write(data)
    except Exception as e:
        log.error(f"Error: {repr(e)}")
        raise e
