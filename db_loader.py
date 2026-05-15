"""DB and id-name map loader for ArtistStyleMatch.

Ports load_embeddings, load_id_name_map, l2_normalize, and resolve_names
from upstream find_artists.py, plus _find_id_name_map for spec §4.3
auto-detection extended with map.json fallback.
"""

import json
import os
from os.path import basename, splitext
from pathlib import Path

import numpy as np


NAME_KEY_CANDIDATES = ("names", "ids", "labels", "artists", "tags", "name", "id")
VECTOR_KEY_CANDIDATES = ("vectors", "embeddings", "features", "vec", "emb")


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def load_id_name_map(path: str) -> dict:
    """Load id->name mapping from a JS or JSON file.

    Supports:
      - data.js with `const xxx = [...];` wrapper (Style Explorer format)
      - plain JSON arrays of objects with 'id' and 'name' fields
      - plain JSON dicts mapping id -> name
    """
    text = Path(path).read_text(encoding="utf-8").strip()

    if text.startswith("const "):
        eq_idx = text.find("=")
        if eq_idx == -1:
            raise ValueError(f"Could not parse JS file {path}: no '=' found")
        text = text[eq_idx + 1:].strip()
        if text.endswith(";"):
            text = text[:-1].strip()

    parsed = json.loads(text)

    mapping: dict = {}
    if isinstance(parsed, list):
        for entry in parsed:
            if isinstance(entry, dict) and "id" in entry and "name" in entry:
                mapping[str(entry["id"])] = entry["name"]
    elif isinstance(parsed, dict):
        for k, v in parsed.items():
            mapping[str(k)] = v if isinstance(v, str) else v.get("name", str(k))
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")

    return mapping


def _find_id_name_map(db_path: str):
    """Auto-detect id-name map next to the .npz (spec §4.3, extended).

    Search order in the same directory as db_path:
      1. <db_stem>_id_name_map.json
      2. id_name_map.json
      3. data.js
      4. map.json   (extension to spec §4.3, user-approved)
    """
    db_dir = os.path.dirname(db_path)
    db_stem = splitext(basename(db_path))[0]
    candidates = [
        f"{db_stem}_id_name_map.json",
        "id_name_map.json",
        "data.js",
        "map.json",
    ]
    for name in candidates:
        p = os.path.join(db_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _load_embeddings(path: str):
    """Load .npz with auto key detection. Ported from find_artists.py:106-150
    with the CLI print() lines removed."""
    data = np.load(path, allow_pickle=True)
    available = list(data.keys())

    name_key = next((k for k in NAME_KEY_CANDIDATES if k in data), None)
    if name_key is None:
        raise ValueError(
            f"{path}: could not auto-detect names key. "
            f"Available: {available}. Tried: {NAME_KEY_CANDIDATES}."
        )

    vector_key = next((k for k in VECTOR_KEY_CANDIDATES if k in data), None)
    if vector_key is None:
        raise ValueError(
            f"{path}: could not auto-detect vectors key. "
            f"Available: {available}. Tried: {VECTOR_KEY_CANDIDATES}."
        )

    names = np.asarray(data[name_key])
    vectors = np.asarray(data[vector_key], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(names):
        raise ValueError(
            f"Shape mismatch: {name_key}={names.shape}, {vector_key}={vectors.shape}"
        )
    return names, vectors


def load_db(db_path: str):
    """Returns (names, vectors_n, id_to_name | None).

    vectors_n is L2-normalized along axis=1. id_to_name is None when no map
    file is found; resolve_names() then falls back to raw IDs.
    """
    names, vectors = _load_embeddings(db_path)
    vectors_n = l2_normalize(vectors, axis=1)

    map_path = _find_id_name_map(db_path)
    id_to_name = load_id_name_map(map_path) if map_path else None

    return names, vectors_n, id_to_name


def resolve_names(id_list, id_to_name) -> list:
    """Resolve raw IDs to human-readable names with multiple normalizations.
    Ported from find_artists.py:487-509 (closure)."""
    if id_to_name is None:
        return [str(i) for i in id_list]
    out = []
    for raw in id_list:
        s = str(raw)
        candidates = [
            s,
            splitext(s)[0],
            basename(s),
            splitext(basename(s))[0],
            s.lstrip("0") or "0",
            str(int(s)) if s.isdigit() else s,
        ]
        resolved = None
        for c in candidates:
            if c in id_to_name:
                resolved = id_to_name[c]
                break
        out.append(resolved if resolved is not None else s)
    return out
