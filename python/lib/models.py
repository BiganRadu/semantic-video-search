"""Model loading, shared by indexing and search.

Both paths must embed into the same space, so the pieces that could silently
diverge -- the tokenizer settings, the normalization -- have exactly one
definition here.

Loading is lazy: search.py needs only the text tower, and on a CPU box the
vision tower is a couple of gigabytes it would never use.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import SIGLIP_MODEL, TEXT_MAX_LEN

log = logging.getLogger(__name__)


def _as_tensor(out):
    """get_*_features returns a bare tensor on transformers 4.x and a pooled
    model output on 5.x. Accept either."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "last_hidden_state"):
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    raise TypeError(f"cannot read features from {type(out).__name__}")


class SigLIP:
    """Image and text into one shared vector space."""

    def __init__(self, device: str = "cpu", *, vision: bool = True):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32

        log.info("loading %s on %s", SIGLIP_MODEL, device)
        model = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=self.dtype).to(device).eval()

        if not vision and getattr(model, "vision_model", None) is not None:
            # Text encoding never calls the vision tower; on a small CPU box
            # those gigabytes matter.
            model.vision_model = None

        self.model = model
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)

    def _normalize(self, feats):
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def embed_images(self, images: list, batch_size: int = 48) -> np.ndarray:
        """One L2-normalized row per image, in order."""
        import torch

        out = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            proc = self.processor(images=chunk, return_tensors="pt")
            proc = {k: v.to(self.device) for k, v in proc.items()}
            with torch.inference_mode():
                feats = _as_tensor(self.model.get_image_features(**proc))
                feats = self._normalize(feats.float())
            out.append(feats.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """One L2-normalized row per text, in order."""
        import torch

        proc = self.processor(
            text=texts,
            padding="max_length",   # see TEXT_MAX_LEN in config
            max_length=TEXT_MAX_LEN,
            truncation=True,
            return_tensors="pt",
        )
        proc = {k: v.to(self.device) for k, v in proc.items()}
        with torch.inference_mode():
            feats = _as_tensor(self.model.get_text_features(**proc))
            feats = self._normalize(feats.float())
        return feats.cpu().numpy()


def vector_literal(vec) -> str:
    """Render a vector in pgvector's text form.

    Text rather than the binary protocol so no type OIDs need registering:
    '%s::vector' and '%s::halfvec' both accept it.
    """
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


class BGEM3:
    """bge-m3 dense embeddings, for text-to-text matching.

    A second embedder alongside SigLIP on purpose: SigLIP is strong at
    image-to-text and weak at text-to-text, so captions and transcripts are
    matched with a model actually trained for it.

    Dense representation is the normalized CLS token, which is what bge-m3's
    own dense retrieval head uses.
    """

    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from .config import BGE_MODEL

        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32

        log.info("loading %s on %s", BGE_MODEL, device)
        self.tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL)
        self.model = AutoModel.from_pretrained(BGE_MODEL, dtype=self.dtype).to(device).eval()

    def embed_texts(self, texts: list[str], batch_size: int = 16,
                    max_length: int = 512) -> np.ndarray:
        import torch

        out = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            proc = self.tokenizer(chunk, padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt")
            proc = {k: v.to(self.device) for k, v in proc.items()}
            with torch.inference_mode():
                hidden = self.model(**proc).last_hidden_state[:, 0]   # CLS
                hidden = hidden.float()
                hidden = hidden / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            out.append(hidden.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)
