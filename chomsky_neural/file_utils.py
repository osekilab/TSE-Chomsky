import json
from pathlib import Path
from typing import Dict, List, Union

import jsonlines


def load_jsonlines(path: Union[str, Path]) -> List[Dict]:
    data_list = []
    with jsonlines.open(str(path)) as reader:
        for data in reader:
            data_list.append(data)
    return data_list


def save_as_jsonlines(
    data: List[Dict],
    path: Union[str, Path],
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with jsonlines.open(str(path), mode="w") as writer:
        for datum in data:
            writer.write(datum)
    return


def load_json(path: Union[str, Path]) -> Dict:
    with open(str(path)) as f:
        data = json.load(f)
    return data


def save_formatted_json(
    data: Union[Dict, List, str],
    path: Union[str, Path],
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    """
    Saves a dictionary or a list which is JSON serializable to a formatted JSON
    (UTF-8, 4 space indent).
    Paramters:
        data: A dictionary or a list which is JSON serializable. JSON data as string
            can also be input.
        path: Path to save the input `data` to.
        parents: Determines whether to make parent directories of the output file.
            Will be input to `pathlib.Path.mkdir` method.
        exist_ok: Determines whether to make parent directory if it exists already.
            Will be input to `pathlib.Path.mkdir` method.
    """
    path = Path(path)
    path.parent.mkdir(parents=parents, exist_ok=exist_ok)
    if isinstance(data, str):
        data = json.loads(data)
    with path.open(mode="w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4, separators=(",", ": "))
    return None
