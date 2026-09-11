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

# Captioner. Qwen publishes no AWQ build of Qwen3-VL-4B and the FP8 build needs
# Ada/Hopper, so the bf16 checkpoint is loaded and quantized to 4-bit NF4 with
# bitsandbytes at load time. That keeps it around 3 GB, which leaves room for
# SigLIP, bge-m3 and Whisper in the same process.
CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
CAPTION_FRAMES = 6        # of the 10 sampled per clip; the throughput dial
CAPTION_MAX_TOKENS = 220
CAPTION_BATCH = 16        # clips per generate() call; a 150s video is 15 clips,
                          # so one batch covers a whole video

# Which stages the indexer currently runs. Stored on each video, so after a
# pipeline change "what needs reindexing?" is a query and not a guess. Bump
# this whenever a stage is added or its output changes meaningfully.
PIPELINE_VERSION = "visual+speech+caption"

# Whisper invents speech in silence -- "Thank you for watching" over music is
# the classic. VAD gates it first, then these thresholds drop what survives.
# Tuned per CLAUDE.md; a dropped segment is far cheaper than a fabricated one
# sitting in the keyword index.
MIN_AVG_LOGPROB = -1.0
MAX_NO_SPEECH_PROB = 0.6
MIN_LANGUAGE_PROB = 0.5

# SigLIP's text tower is trained with a fixed 64-token context. Any padding
# other than to max_length silently produces wrong embeddings: the model does
# not error, it returns vectors from a distribution it never saw.
TEXT_MAX_LEN = 64

# --- runtime ---------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "postgres://video:video@localhost:5433/video")
SCRATCH = Path(os.environ.get("SCRATCH_DIR", ROOT / "scratch"))

# Longest video the add-video path will accept, in seconds. Indexing costs
# roughly a minute of GPU time per minute of video, so without a cap one
# submitted lecture recording occupies the machine for an afternoon. 0 disables.
MAX_SOURCE_DURATION = float(os.environ.get("MAX_SOURCE_DURATION", 60 * 60))


def device_for(role: str) -> str:
    """Indexing wants the GPU; search must work without one."""
    import torch

    override = os.environ.get(f"{role.upper()}_DEVICE")
    if override:
        return override
    if role == "index" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
