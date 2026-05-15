"""ArtistStyleMatch ComfyUI node."""

import os
import threading
from math import comb

import numpy as np

from . import db_loader, encoding, search


def _comfy_image_to_pil(image_tensor):
    """ComfyUI IMAGE (B, H, W, C float32 [0,1]) -> PIL.Image (mode RGB)."""
    from PIL import Image
    arr = (image_tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


class ArtistStyleMatch:
    # Class-level caches (spec §3.5: instance is recreated per graph run).
    _encoder_cache: dict = {}
    _db_cache: dict = {}
    _cache_lock = threading.Lock()

    @classmethod
    def _list_db_files(cls):
        import folder_paths
        try:
            return folder_paths.get_filename_list("artist_embeddings")
        except Exception:
            return []

    @classmethod
    def INPUT_TYPES(cls):
        db_files = cls._list_db_files()
        return {
            "required": {
                "image": ("IMAGE",),
                "db_file": (
                    db_files
                    if db_files
                    else ["<no .npz in models/artist_embeddings/>"],
                ),
                "model_type": (["dinov2", "clip"],),
                "k": ("INT", {"default": 3, "min": 1, "max": 5}),
                "top_n": ("INT", {"default": 200, "min": 50, "max": 2000, "step": 50}),
                "max_weight": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.5, "max": 2.0, "step": 0.05},
                ),
                "clip_model_name": (
                    "STRING",
                    {"default": "openai/clip-vit-large-patch14"},
                ),
                "weight_rescale": (
                    ["max", "sum", "none", "top1_anchor"],
                    {"default": "max"},
                ),
                "rescale_target": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05},
                ),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("prompt", "info", "cosine_similarity")
    FUNCTION = "execute"
    CATEGORY = "Artist Search"

    @classmethod
    def _get_encoder(cls, model_type, clip_model_name, device):
        # Double-check insert: lookup -> unlock -> load -> re-lock -> insert.
        key = (
            model_type,
            clip_model_name if model_type == "clip" else None,
            device,
        )
        with cls._cache_lock:
            cached = cls._encoder_cache.get(key)
            if cached is not None:
                return cached

        loaded = encoding.load_encoder(model_type, clip_model_name, device)

        with cls._cache_lock:
            existing = cls._encoder_cache.get(key)
            if existing is not None:
                return existing
            cls._encoder_cache[key] = loaded
            return loaded

    @classmethod
    def _get_db(cls, db_path):
        # Cache key includes db mtime AND id-name map mtime so map-only swaps
        # are also detected (Plan §決定事項).
        db_mtime = os.path.getmtime(db_path)
        map_path = db_loader._find_id_name_map(db_path)
        map_mtime = os.path.getmtime(map_path) if map_path else 0.0
        key = (db_path, db_mtime, map_path or "", map_mtime)

        with cls._cache_lock:
            cached = cls._db_cache.get(key)
            if cached is not None:
                return cached
            # Drop any prior entry for the same db_path (different mtime).
            for stale_key in [k for k in cls._db_cache if k[0] == db_path]:
                cls._db_cache.pop(stale_key, None)

        loaded = db_loader.load_db(db_path)

        with cls._cache_lock:
            existing = cls._db_cache.get(key)
            if existing is not None:
                return existing
            cls._db_cache[key] = loaded
            return loaded

    def execute(
        self,
        image,
        db_file,
        model_type,
        k,
        top_n,
        max_weight,
        clip_model_name,
        weight_rescale,
        rescale_target,
        scale,
    ):
        import folder_paths

        # 1. Image conversion (spec §3.2). Batch > 1 -> warn and use [0].
        if hasattr(image, "shape") and image.shape[0] > 1:
            print(
                f"[ArtistStyleMatch] warning: batch size {image.shape[0]} > 1, "
                f"using image[0] only"
            )
        pil = _comfy_image_to_pil(image)

        # 2. Device (spec §6.4).
        try:
            import comfy.model_management as mm
            device = str(mm.get_torch_device())
        except Exception:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 3. Resolve DB path. None / sentinel -> spec §4.4 error.
        if not db_file or db_file.startswith("<no "):
            raise FileNotFoundError(
                "No DB file selected. Place an .npz embeddings file under "
                "ComfyUI/models/artist_embeddings/ and restart ComfyUI."
            )
        db_path = folder_paths.get_full_path("artist_embeddings", db_file)
        if db_path is None or not os.path.isfile(db_path):
            raise FileNotFoundError(
                f"DB file not found: {db_file}. "
                f"Place .npz files under ComfyUI/models/artist_embeddings/."
            )

        # 4-5. Cache lookups.
        names, vectors_n, id_to_name = self._get_db(db_path)
        model, processor, effective_device = self._get_encoder(
            model_type, clip_model_name, device
        )

        # 6. Encode.
        target = encoding.encode_image(
            model, processor, model_type, pil, effective_device
        )

        # 7. Dim check (spec §4.4).
        if target.shape[0] != vectors_n.shape[1]:
            raise ValueError(
                f"Dim mismatch: encoded target dim={target.shape[0]} vs "
                f"DB dim={vectors_n.shape[1]} (db_file={db_file}). "
                f"Make sure model_type ({model_type}) matches the encoder "
                f"used to build the DB. For CLIP, also check clip_model_name "
                f"(current: {clip_model_name}). Common dims: CLIP ViT-L w/ "
                f"projection=768, w/o projection=1024; CLIP ViT-H=1024; "
                f"DINOv2-large=1024; DINOv2-giant=1536."
            )

        # 8. Normalize target.
        target_n = target / (np.linalg.norm(target) + 1e-12)

        # 9. ProgressBar — total must equal the number of cb() invocations
        # in search_combinations' BVLS refine loop.
        refine_top = 200
        pool_n = min(top_n, len(names))
        num_combos = comb(pool_n, k) if 1 <= k <= pool_n else 0
        total = 0 if k == 1 else min(refine_top, num_combos)
        cb = None
        if total > 0:
            try:
                from comfy.utils import ProgressBar
                pbar = ProgressBar(total)
                cb = lambda: pbar.update(1)
            except Exception:
                cb = None

        # 10. Search.
        results = search.search(
            target_n=target_n,
            db_vectors_n=vectors_n,
            db_names=names,
            k=k,
            top_n=top_n,
            max_weight=max_weight,
            refine_top=refine_top,
            progress_callback=cb,
        )

        # 10b. top1_anchor needs an auxiliary k=1 search. The prefilter
        # inside search.search is stateless (argpartition + sort), so the
        # candidate pool is identical to the k>1 call.
        top1_weight = None
        if weight_rescale == "top1_anchor":
            top1_results = search.search(
                target_n=target_n,
                db_vectors_n=vectors_n,
                db_names=names,
                k=1,
                top_n=top_n,
                max_weight=max_weight,
                refine_top=1,
                progress_callback=None,
            )
            if top1_results and top1_results[0]["weights"]:
                top1_weight = float(top1_results[0]["weights"][0])
            else:
                top1_weight = 0.0

        # 10c. Rescale.
        results = search.rescale_weights(
            results,
            mode=weight_rescale,
            target=rescale_target,
            scale=scale,
            top1_weight=top1_weight,
            max_output=2.0,
        )

        # 11. Resolve human-readable names.
        for r in results:
            r["artist_names"] = db_loader.resolve_names(r["names"], id_to_name)

        # 12. Format outputs.
        if not results:
            return ("", "(no results)", 0.0)
        top = results[0]
        prompt = search.format_prompt(top["artist_names"], top["weights"])
        info = self._format_info(
            results,
            mode=weight_rescale,
            target=rescale_target,
            scale=scale,
            top1_weight=top1_weight,
            top_k=5,
        )
        cos = float(top["cosine_similarity"])
        return (prompt, info, cos)

    @staticmethod
    def _format_info(results, mode, target, scale, top1_weight=None, top_k=5):
        n = min(top_k, len(results))
        lines = [f"Rescale: {mode}(target={target:.2f}) × scale={scale:.2f}"]
        if mode == "top1_anchor" and top1_weight is not None:
            lines.append(f"top1_weight={top1_weight:.4f}")
        lines.append(f"Top {n} combinations (cos similarity):")
        for i, r in enumerate(results[:n]):
            frag = search.format_prompt(r["artist_names"], r["weights"])
            lines.append(f"  {i+1}. {r['cosine_similarity']:.4f}  prompt: {frag}")
            raw = r.get("raw_weights")
            if raw is not None:
                raw_str = ", ".join(f"{w:.2f}" for w in raw)
                lines.append(f"             raw: {raw_str}")
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {"ArtistStyleMatch": ArtistStyleMatch}
NODE_DISPLAY_NAME_MAPPINGS = {"ArtistStyleMatch": "Artist Style Match"}
