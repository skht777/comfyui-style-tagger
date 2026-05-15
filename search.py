"""Combination search core for ArtistStyleMatch.

Ports search_combinations and format_prompt from find_artists.py, plus a
prefilter+search wrapper. Algorithm is verified per spec §6.1 — do not alter
the closed-form batch solve or the Gram-only cosine evaluation.
"""

from itertools import combinations

import numpy as np


def search(
    target_n: np.ndarray,
    db_vectors_n: np.ndarray,
    db_names: np.ndarray,
    k: int,
    top_n: int,
    max_weight: float,
    refine_top: int = 200,
    progress_callback=None,
) -> list:
    """Prefilter to top_n candidates, then enumerate k-combinations.

    db_vectors_n must already be L2-normalized along axis=1.
    Returns sorted list of {indices, names, weights, cosine_similarity}.
    """
    pool_n = min(top_n, len(db_names))
    sims_all = db_vectors_n @ target_n
    top_idx = np.argpartition(-sims_all, pool_n - 1)[:pool_n]
    top_idx = top_idx[np.argsort(-sims_all[top_idx])]

    return search_combinations(
        target_n=target_n,
        cand_vectors=db_vectors_n[top_idx],
        cand_indices=top_idx,
        cand_names=db_names[top_idx],
        k=k,
        max_weight=max_weight,
        refine_top=refine_top,
        progress_callback=progress_callback,
    )


def search_combinations(
    target_n,
    cand_vectors,
    cand_indices,
    cand_names,
    k,
    max_weight,
    refine_top,
    progress_callback=None,
):
    """Vectorized closed-form sweep over all C(N, k), then refine top R with BVLS.

    Ported from find_artists.py:239-347. Algorithm verified per spec §6.1.
    Differences from upstream: print() removed; progress_callback() invoked
    once per BVLS refine iteration so the ComfyUI progress bar updates.
    """
    n = len(cand_vectors)
    if k < 1 or k > n:
        raise ValueError(f"Invalid k={k} for pool size n={n}")

    if k == 1:
        scores = cand_vectors @ target_n
        weights = np.clip(scores, 0.0, max_weight)
        order = np.argsort(-scores)
        results = []
        for j in order:
            w = float(weights[j])
            cos = float(scores[j]) if w > 0 else 0.0
            results.append({
                "indices": [int(cand_indices[j])],
                "names": [str(cand_names[j])],
                "weights": [w],
                "cosine_similarity": cos,
            })
        return results

    G_full = cand_vectors @ cand_vectors.T
    b_full = cand_vectors @ target_n

    combos = np.fromiter(
        (i for combo in combinations(range(n), k) for i in combo),
        dtype=np.int32,
    ).reshape(-1, k)
    num_combos = combos.shape[0]

    batch_G = G_full[combos[:, :, None], combos[:, None, :]]
    batch_b = b_full[combos]

    eye_k = np.eye(k, dtype=batch_G.dtype) * 1e-8
    try:
        batch_w_uncon = np.linalg.solve(
            batch_G + eye_k, batch_b[..., None]
        ).squeeze(-1)
    except np.linalg.LinAlgError:
        batch_w_uncon = np.zeros((num_combos, k), dtype=np.float32)
        for i in range(num_combos):
            try:
                batch_w_uncon[i] = np.linalg.solve(batch_G[i] + eye_k, batch_b[i])
            except np.linalg.LinAlgError:
                batch_w_uncon[i] = np.linalg.lstsq(
                    batch_G[i], batch_b[i], rcond=None
                )[0]

    batch_w = np.clip(batch_w_uncon, 0.0, max_weight)

    numerator = np.sum(batch_w * batch_b, axis=1)
    quad = np.einsum("ij,ijk,ik->i", batch_w, batch_G, batch_w)
    denom = np.sqrt(np.maximum(quad, 0.0)) + 1e-12
    cos_sims = numerator / denom

    refine_n = min(refine_top, num_combos)
    top_idx = np.argpartition(-cos_sims, refine_n - 1)[:refine_n]
    top_idx = top_idx[np.argsort(-cos_sims[top_idx])]

    from scipy.optimize import lsq_linear

    refined = []
    for idx in top_idx:
        combo = combos[idx]
        V = cand_vectors[combo]
        res = lsq_linear(V.T, target_n, bounds=(0.0, max_weight), method="bvls")
        w = res.x.astype(np.float32)
        combined = V.T @ w
        nrm = np.linalg.norm(combined) + 1e-12
        cos = float((combined @ target_n) / nrm)
        refined.append({
            "indices": [int(cand_indices[i]) for i in combo],
            "names":   [str(cand_names[i])   for i in combo],
            "weights": [float(x) for x in w],
            "cosine_similarity": cos,
        })
        if progress_callback is not None:
            progress_callback()

    refined.sort(key=lambda r: -r["cosine_similarity"])
    return refined


def rescale_weights(
    results,
    mode,
    target,
    scale=1.0,
    top1_weight=None,
    max_output=2.0,
):
    """Post-process each result's weights in-place.

    For each result:
      - 'raw_weights' <- original raw weights (preserved on re-application
                         so calling this twice does not double-rescale).
      - 'prompt_weights' / 'weights' <- rescale(raw, mode, target) * scale.

    If any rescaled weight exceeds max_output, a single aggregated warning
    is printed (count + worst value) — never per-result.
    """
    n_warn = 0
    worst = 0.0
    for r in results:
        raw_source = r.get("raw_weights", r["weights"])
        raw = np.array(raw_source, dtype=np.float64)
        if mode == "none":
            new = raw.copy()
        elif mode == "max":
            mx = raw.max() if raw.size else 0.0
            new = raw * (target / mx) if mx > 1e-9 else raw.copy()
        elif mode == "sum":
            s = raw.sum()
            new = raw * (target / s) if s > 1e-9 else raw.copy()
        elif mode == "top1_anchor":
            if top1_weight is None:
                raise ValueError("top1_anchor mode requires top1_weight")
            s = raw.sum()
            new = raw * (top1_weight / s) if s > 1e-9 else raw.copy()
        else:
            raise ValueError(f"Unknown rescale mode: {mode}")

        new = new * scale

        r["raw_weights"] = raw.tolist()
        r["prompt_weights"] = new.tolist()
        r["weights"] = new.tolist()

        if new.size and new.max() > max_output:
            n_warn += 1
            worst = max(worst, float(new.max()))

    if n_warn:
        print(
            f"[ArtistStyleMatch] warning: {n_warn} result(s) have "
            f"rescaled weight up to {worst:.2f} (exceeds {max_output}); "
            f"not clipping."
        )
    return results


def format_prompt(names, weights, threshold: float = 0.05) -> str:
    """Build SD-style prompt fragment, dropping near-zero weights.
    Ported from find_artists.py:354-361."""
    parts = []
    for n, w in zip(names, weights):
        if w < threshold:
            continue
        parts.append(f"({n}:{w:.2f})")
    return ", ".join(parts) if parts else "(no significant weights)"
