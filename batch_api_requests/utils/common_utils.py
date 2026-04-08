"""Common utility functions for batch API requests."""
import json
import pathlib
import io
from typing import List, Dict, Any, Union

PathOrIOBase = Union[str, pathlib.Path, io.IOBase]

def load_text_lines(filepath: str) -> List[Dict[str, str]]:
    """Load a text file and return list of dicts with 'text' key."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append({"text": line.strip()})
    return data

def readlines(f: Union[str, pathlib.Path, io.IOBase], mode="r", strip=True):
    f = _make_r_io_base(f, mode)
    lines = f.readlines()
    if strip:
        lines = [line.strip() for line in lines]
    f.close()
    return lines

def jlload(f: PathOrIOBase, mode="r", strip=True):
    """Load a .jsonl file into a list of dictionaries."""
    return [json.loads(line) for line in readlines(f, mode=mode, strip=strip)]

def read(f: Union[str, pathlib.Path, io.IOBase], mode="r", strip=True):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    content = f.read()
    if strip:
        content = content.strip()
    f.close()
    return content

def _make_r_io_base(f: PathOrIOBase, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f: PathOrIOBase, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
