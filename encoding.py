"""Image encoder for ArtistStyleMatch (CLIP / DINOv2).

Split from find_artists.py:encode_target into load + encode so the model
can be cached across inferences (the upstream gc/empty_cache cleanup in the
finally block at lines 218-232 is intentionally NOT ported, since the model
must stay resident).
"""

import numpy as np


def load_encoder(model_type: str, clip_model_name: str, device: str):
    """Load model + processor. Returns (model, processor, effective_device).

    Includes the CUDA-unavailable fallback from find_artists.py:171-174
    (spec §6.4): if device starts with "cuda" but torch.cuda.is_available()
    is False, fall back to "cpu" with a warning. The actually-used device is
    returned so the caller can pass it to encode_image.
    """
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[ArtistStyleMatch] warning: {device} not available, "
            f"falling back to cpu"
        )
        device = "cpu"

    if model_type == "clip":
        from transformers import CLIPModel, CLIPProcessor
        processor = CLIPProcessor.from_pretrained(clip_model_name)
        model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    elif model_type == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        model_name = "facebook/dinov2-large"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, processor, device


def encode_image(model, processor, model_type: str, pil_image, device: str) -> np.ndarray:
    """Encode one PIL image to a 1D float32 vector (NOT L2-normalized).

    CLIP branch includes the BaseModelOutputWithPooling fallback (spec §6.2,
    find_artists.py:187-198): some transformers versions/configs return a
    ModelOutput object instead of a Tensor.
    """
    import torch

    with torch.no_grad():
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        if model_type == "clip":
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                if getattr(features, "image_embeds", None) is not None:
                    features = features.image_embeds
                elif getattr(features, "pooler_output", None) is not None:
                    features = features.pooler_output
                elif getattr(features, "last_hidden_state", None) is not None:
                    features = features.last_hidden_state[:, 0, :]
                else:
                    raise RuntimeError(
                        f"Unexpected CLIP output type: {type(features).__name__}. "
                        f"Cannot extract a tensor."
                    )
        elif model_type == "dinov2":
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    return features.cpu().numpy().squeeze().astype(np.float32)
