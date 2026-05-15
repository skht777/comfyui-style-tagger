"""ArtistStyleMatch ComfyUI custom node entry point.

Registers `models/artist_embeddings/` with folder_paths and re-exports the
node class mappings.
"""

import os

import folder_paths

_db_dir = os.path.join(folder_paths.models_dir, "artist_embeddings")
os.makedirs(_db_dir, exist_ok=True)
if "artist_embeddings" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["artist_embeddings"] = (
        [_db_dir],
        {".npz"},
    )
else:
    existing_paths, existing_exts = folder_paths.folder_names_and_paths[
        "artist_embeddings"
    ]
    if _db_dir not in existing_paths:
        existing_paths.append(_db_dir)
    existing_exts.add(".npz")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
