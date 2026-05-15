# comfyui-artist-match

ComfyUI custom node `ArtistStyleMatch` — given a reference image, encode it
with CLIP or DINOv2, search a pre-built artist embedding `.npz`, and emit a
weighted Stable Diffusion prompt fragment of the form
`(artist1:weight1), (artist2:weight2), ...`.

This is a port of the upstream CLI tool
[`find_artists.py`](https://github.com/skht777/Illustrious-NoobAI-Style-Explorer/blob/main/scripts/find_artists.py)
into a single ComfyUI node. The search algorithm (closed-form combination sweep + bounded NNLS refine) is reused verbatim.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone <this-repo-url> comfyui-artist-search
pip install -r comfyui-artist-search/requirements.txt
```

> Note: ComfyUI environments usually already provide compatible `torch` and
> `numpy`. If pip wants to reinstall them, prefer `--no-deps` or install
> only the missing packages (`scipy`, `transformers`, `pillow`).

Restart ComfyUI. The node appears under `Artist Search/Artist Style Match`.

## DB setup

Place the embedding DB into `ComfyUI/models/artist_embeddings/`:

```
ComfyUI/models/artist_embeddings/
├── embeddings_dinov2.npz      (required)
├── id_name_map.json           (recommended — see below)
└── embeddings_clip.npz        (optional, for CLIP encoder)
```

`.npz` files come from the upstream
[Illustrious-NoobAI-Style-Explorer](https://github.com/skht777/Illustrious-NoobAI-Style-Explorer)
repo's `data/` directory.

For human-readable artist names, also place an id-name map. The node
auto-detects, in order:

1. `<db_stem>_id_name_map.json`  (e.g. `embeddings_dinov2_id_name_map.json`)
2. `id_name_map.json`
3. `data.js`
4. `map.json`  (upstream raw metadata file — works as-is)

To generate `id_name_map.json` from `data.js`, run the upstream
`build_id_name_map.py`:

```bash
python Illustrious-NoobAI-Style-Explorer/scripts/build_id_name_map.py \
    data.js -o id_name_map.json
```

If no map is found, the node falls back to raw IDs in the prompt.

## Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | (required) | Reference image |
| `db_file` | dropdown (.npz) | first | DB file in `models/artist_embeddings/` |
| `model_type` | `["dinov2", "clip"]` | dinov2 | Encoder — must match the DB |
| `k` | INT 1-5 | 3 | Number of artists to combine |
| `top_n` | INT 50-2000 step 50 | 200 | Pre-filter pool size |
| `max_weight` | FLOAT 0.5-2.0 step 0.05 | 1.5 | **Search-time cap on raw weights** (applied inside BVLS). Not the upper bound on the rescaled prompt weights — those can exceed this. |
| `clip_model_name` | STRING | `openai/clip-vit-large-patch14` | CLIP model id |
| `weight_rescale` | `["max", "sum", "none", "top1_anchor"]` | `max` | Post-search rescaling. `max`: top weight → `rescale_target`. `sum`: total → `rescale_target` (recommended target 1.5). `none`: raw embedding solution. `top1_anchor`: use the k=1 optimal weight as total strength. |
| `rescale_target` | FLOAT 0.1-3.0 step 0.05 | 1.0 | Target value for `weight_rescale`. Use 1.0 for `max`, ~1.5 for `sum`. Ignored for `none` / `top1_anchor`. |
| `scale` | FLOAT 0.0-2.0 step 0.05 | 1.0 | Uniform multiplier applied **after** rescale (all modes, including `none`). <1.0 weakens, >1.0 strengthens. `scale=0` yields an empty-effect prompt. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `prompt` | STRING | Top-1 combination as `(name:weight), ...` (weights are post-rescale) |
| `info` | STRING | Mode header + top-5 candidates with cosine similarity, prompt weights, and raw weights |
| `cosine_similarity` | FLOAT | Top-1 cosine similarity. Computed at search time on raw embeddings, so it is **unchanged** by `weight_rescale` / `rescale_target` / `scale`. |

> Note: `raw_weights` and `prompt_weights` are internal fields (visible in
> the `info` string), not separate node outputs. The public RETURN is
> still the three values above.

## Behavior change (vs initial release)

The `prompt` output now reflects **rescaled** weights by default
(`weight_rescale=max`, `rescale_target=1.0`, `scale=1.0`). To recover the
prior behavior that emitted raw embedding-space weights, set
`weight_rescale="none"` and `scale=1.0`.

- Relationship: `prompt_weight = rescale(raw_weight, mode, target) × scale`.
- `cosine_similarity` is computed from the raw search and does **not**
  depend on the rescale settings — switching modes does not change it.
- `scale=0` produces `(no significant weights)` in the prompt (all rescaled
  weights are zero), while `cosine_similarity` still shows the raw match.
- Uniform scaling (α > 0) does not change cosine similarity, so the
  ranking among the top-K is preserved across all modes / scale values
  (mathematical fact, not coincidence).

## Modes — quick guide

- `max` (default): set the top weight to `rescale_target` (e.g. 1.0).
  Most natural for SD prompts — the leading artist gets standard strength.
- `sum`: make the weights add up to `rescale_target` (recommended ~1.5).
  Keeps total "style mass" constant; gentler than `max`.
- `none`: pass raw weights through. For analysis / comparison.
- `top1_anchor`: rerun the search with k=1 and use its best weight as
  the total strength. The `info` line shows this `top1_weight`.

## Example graph

```
Load Image ──► Artist Style Match ──► ShowText (prompt)
                                  └─► ShowText (info)
                                  └─► (FLOAT) cosine_similarity
```

Pipe the `prompt` output into a CLIP Text Encode node to use it for
generation.

## DINOv2 vs CLIP

- **DINOv2** (default): captures visual style well; usually preferred for
  style transfer.
- **CLIP**: also encodes semantic content. Try this if DINOv2 results lean
  too heavily on raw visual texture.

`model_type` must match the encoder used to build the DB. The dim-mismatch
error message (see Troubleshooting) tells you the expected vs actual dim.

## Troubleshooting

- **`No DB file selected` / `DB file not found`**: drop a `.npz` into
  `ComfyUI/models/artist_embeddings/` and restart ComfyUI.
- **`Dim mismatch: encoded target dim=X vs DB dim=Y`**: `model_type` does
  not match the DB. For CLIP, also verify `clip_model_name`. Common dims:
  CLIP ViT-L w/ projection=768, w/o projection=1024; CLIP ViT-H=1024;
  DINOv2-large=1024; DINOv2-giant=1536.
- **First run is slow (~30s)**: HuggingFace is downloading the encoder
  (~1.5GB). Subsequent runs hit the cache and complete in ~5s.
- **Prompt shows raw numeric IDs**: no id-name map found, or IDs do not
  resolve. Place a map (see DB setup) and retry.
- **Old workflow shows wrong values after upgrade**: The rescale widgets
  (`weight_rescale`, `rescale_target`, `scale`) were added at the end of
  the input list specifically to preserve widget-order compatibility, so
  existing saved workflows should reload with defaults. If you see other
  widgets (e.g. `clip_model_name`) suddenly holding numeric values, the
  workflow may have been saved against an intermediate version — delete
  and re-add the node.

## v2 ideas (not implemented)

- Mask input — extract style from a region only
- Multi-reference image averaging
- Multiple top-K outputs as separate strings
- `uniqueness_score` filter
- Node split (LoadDB / Encode / Search / Format)
