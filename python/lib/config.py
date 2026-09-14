"""Settings shared by index.py and search.py."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- segmentation ----------------------------------------------------------
CLIP_LEN = 10.0     # seconds per clip
FRAME_RATE = 1.0    # frames sampled per second
FRAME_SIZE = 384    # SigLIP 2 so400m-patch14-384 input edge

# --- models ----------------------------------------------------------------
SIGLIP_MODEL = os.environ.get("SIGLIP_MODEL", "google/siglip2-so400m-patch14-384")
VISUAL_DIM = 1152   # must match clip_vectors.visual in the schema

BGE_MODEL = os.environ.get("BGE_MODEL", "BAAI/bge-m3")
TEXT_DIM = 1024     # must match clip_vectors.caption_vec / speech_vec

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

# Loaded in 4-bit at runtime unless the checkpoint is already quantized.
CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
CAPTION_FRAMES = 6        # of the 10 sampled per clip; the throughput dial
CAPTION_MAX_TOKENS = 320  # a ceiling, not a cost: generation stops at EOS
# Clips per generate() call. Sized for a 24 GB card; lower it on a smaller one.
CAPTION_BATCH = int(os.environ.get("CAPTION_BATCH", 16))

# Which stages the indexer runs, stored on each video so "what needs
# reindexing?" is a query rather than a guess. Bump it when a stage is added or
# its output changes meaningfully.
PIPELINE_VERSION = "visual+speech+caption+c2"

# Whisper invents speech in silence. VAD gates it first; these drop what
# survives. A dropped segment is cheaper than a fabricated one in the index.
MIN_AVG_LOGPROB = -1.0
MAX_NO_SPEECH_PROB = 0.6
MIN_LANGUAGE_PROB = 0.5

# SigLIP's text tower has a fixed 64-token context. Padding to anything else
# silently produces wrong embeddings rather than an error.
TEXT_MAX_LEN = 64

# --- runtime ---------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "postgres://video:video@localhost:5433/video")
SCRATCH = Path(os.environ.get("SCRATCH_DIR", ROOT / "scratch"))

# Longest video accepted for indexing, which costs roughly a minute of GPU per
# minute of video. 0 disables the cap.
MAX_SOURCE_DURATION = float(os.environ.get("MAX_SOURCE_DURATION", 60 * 60))

# --- query planning --------------------------------------------------------
# A model classifies each query and rephrases it per index; the class picks the
# fusion weights. Optional: with no key, search runs on the default weights.
# The call is synchronous, so latency matters more than accuracy here.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
INTENT_BASE_URL = os.environ.get(
    "INTENT_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
# A moving alias; pin a version if drift would matter.
INTENT_MODEL = os.environ.get("INTENT_MODEL", "gemini-flash-lite-latest")
# Bounds a hung connection rather than racing a slow model. On timeout the
# search proceeds with the raw query and default weights.
INTENT_TIMEOUT = float(os.environ.get("INTENT_TIMEOUT", 10))


def device_for(role: str) -> str:
    """Indexing wants the GPU; search must work without one."""
    import torch

    override = os.environ.get(f"{role.upper()}_DEVICE")
    if override:
        return override
    if role == "index" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
