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
# Must clear the schema ceiling with room to spare: if the budget runs out
# before the JSON object closes, the caption is unparseable and the clip loses
# it entirely. Worst observed payload across 541 clips is ~690 chars, about 175
# tokens, so 220 was one verbose tag list away from silently dropping a caption.
# This is a ceiling, not a cost -- generation still stops at EOS.
CAPTION_MAX_TOKENS = 320
# Clips per generate() call; a 150s video is 15 clips, so one batch covers a
# whole video. Overridable because 16 is tuned for a 24 GB card: on a 15 GB
# Kaggle T4, holding SigLIP, bge-m3, Whisper and the 4-bit captioner at once
# leaves too little for a batch that size, and it dies part-way through
# captioning rather than at load.
CAPTION_BATCH = int(os.environ.get("CAPTION_BATCH", 16))

# Which stages the indexer currently runs. Stored on each video, so after a
# pipeline change "what needs reindexing?" is a query and not a guess. Bump
# this whenever a stage is added or its output changes meaningfully.
# +c2: caption prompt and post-processing changed (no frame-by-frame walk,
# severed tails closed off). Clips indexed before it hold captions cut
# mid-word, so this is what separates them from the repaired ones.
PIPELINE_VERSION = "visual+speech+caption+c2"

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

# --- query planning --------------------------------------------------------
# A model classifies each query and rephrases it for each index; the class picks
# the fusion weights. Optional: with no key, search runs exactly as before.
#
# Gemini flash-lite, reached through its OpenAI-compatible endpoint. The call is
# synchronous and uncached, so latency is the whole selection criterion -- it is
# time a user spends staring at a spinner. Measured on six queries with this
# prompt: 6/6 correct, p50 742ms, range 646-897ms. A previously supported
# provider (DeepSeek on NVIDIA NIM) classified 7/8 but ranged 0.4s to 182s for
# identical work, which is not a search feature; it was removed rather than kept
# as a fallback nobody would want to fall back to.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
INTENT_BASE_URL = os.environ.get(
    "INTENT_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
# A moving alias: Google decides what it points at. Pin it (gemini-3.5-flash-lite)
# if the classifier's behaviour changing underneath you would matter.
INTENT_MODEL = os.environ.get("INTENT_MODEL", "gemini-flash-lite-latest")

# A search waits for this, so the timeout is the worst case a user can feel.
# Generous against a ~900ms p99 -- it bounds a hung connection, it does not race
# a slow model. On timeout the search proceeds with the raw query and the
# measured default weights.
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
