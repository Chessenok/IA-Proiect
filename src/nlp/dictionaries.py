from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


def _nlp_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "nlp"


@lru_cache(maxsize=1)
def load_all_dictionaries() -> dict[str, dict[str, Any]]:
    """Încarcă toate dicționarele JSON din data/nlp/."""
    result: dict[str, dict[str, Any]] = {}
    data_dir = _nlp_data_dir()
    if not data_dir.is_dir():
        return result

    for path in sorted(data_dir.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        name = data.get("name", path.stem)
        result[name] = data
    return result


def list_dictionaries() -> list[str]:
    return list(load_all_dictionaries().keys())


def get_dictionary(name: str) -> dict[str, Any] | None:
    return load_all_dictionaries().get(name)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZăâîșțĂÂÎȘȚ]+", text.lower())
