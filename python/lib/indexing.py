"""Models used only when indexing.

Kept apart from lib/models.py so the search path's import graph never reaches
Whisper or the captioner. A machine serving search should not need faster-whisper
installed at all, and scripts/check-boundary.sh enforces that.
"""
from __future__ import annotations

import logging

from .models import BGEM3, SigLIP

log = logging.getLogger(__name__)


class Transcriber:
    """faster-whisper with Silero VAD in front of it.

    task="translate" so non-English audio lands in English alongside the
    captions; the detected language is kept as a filter field rather than
    discarded.
    """

    def __init__(self, device: str = "cpu"):
        from faster_whisper import WhisperModel

        from .config import WHISPER_MODEL

        compute_type = "int8_float16" if device.startswith("cuda") else "int8"
        log.info("loading whisper %s on %s (%s)", WHISPER_MODEL, device, compute_type)
        self.model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)

    def transcribe(self, path) -> tuple[list[dict], list[dict], str | None]:
        """Return (segments, words, detected_language).

        Two granularities, because they serve different purposes. Whisper emits
        ~30s segments, which read well in a transcript panel but are three times
        coarser than a 10s clip -- attributing a whole segment to a clip would
        give three consecutive clips identical speech and identical vectors.

        So word timings drive the clip mapping, and the segments are kept as-is
        for display.
        """
        from .config import (MAX_NO_SPEECH_PROB, MIN_AVG_LOGPROB,
                             MIN_LANGUAGE_PROB)

        segments, info = self.model.transcribe(
            str(path),
            task="translate",
            vad_filter=True,                      # Silero, bundled
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,     # stops hallucination cascades
            word_timestamps=True,                 # needed for clip-level mapping
        )
        language = getattr(info, "language", None)

        if getattr(info, "language_probability", 1.0) < MIN_LANGUAGE_PROB:
            # Language detection this unsure usually means there is no speech.
            return [], [], language

        kept, words = [], []
        for seg in segments:
            if seg.avg_logprob < MIN_AVG_LOGPROB:
                continue
            if seg.no_speech_prob > MAX_NO_SPEECH_PROB:
                continue
            text = (seg.text or "").strip()
            if not text:
                continue
            kept.append({
                "start_s": float(seg.start),
                "end_s": float(seg.end),
                "text": text,
                "avg_logprob": float(seg.avg_logprob),
                "no_speech_prob": float(seg.no_speech_prob),
            })
            for w in (seg.words or []):
                token = (w.word or "").strip()
                if token:
                    words.append({"start_s": float(w.start),
                                  "end_s": float(w.end),
                                  "text": token})
        return kept, words, language


def speech_for_clips(clips, words: list[dict]) -> list[str]:
    """Assign spoken words to the clip they were spoken in.

    Driven by word timings rather than Whisper's ~30s segments, so a 10s clip
    gets the words actually said during it. A word straddling a boundary goes
    to the clip holding most of it.
    """
    out = []
    for clip in clips:
        spoken = [
            w["text"] for w in words
            if min(w["end_s"], clip.end_s) - max(w["start_s"], clip.start_s)
            > (w["end_s"] - w["start_s"]) / 2
        ]
        out.append(" ".join(spoken).strip())
    return out


class IndexModels:
    """The models indexing needs, loaded lazily and shared across videos.

    One instance per process: loading these costs far more than running them on
    a single video, so the bulk script builds one and reuses it for hundreds.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._siglip = None
        self._bge = None
        self._whisper = None
        self._captioner = None

    @property
    def siglip(self) -> "SigLIP":
        if self._siglip is None:
            self._siglip = SigLIP(self.device)
        return self._siglip

    @property
    def bge(self) -> "BGEM3":
        if self._bge is None:
            self._bge = BGEM3(self.device)
        return self._bge

    @property
    def whisper(self) -> "Transcriber":
        if self._whisper is None:
            self._whisper = Transcriber(self.device)
        return self._whisper

    @property
    def captioner(self) -> "Captioner":
        if self._captioner is None:
            self._captioner = Captioner(self.device)
        return self._captioner


# Lengths are part of the schema, not just the prompt. Constrained decoding
# blocks EOS until the object is complete, so an unbounded string field means
# the model elaborates until it hits the token limit and the JSON never closes.
# Bounding the fields is what makes the output both valid and cheap.
_SHORT_TEXT = {"type": "string", "maxLength": 90}

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string", "maxLength": 260},
        "people": {"type": "array", "items": _SHORT_TEXT, "maxItems": 5},
        "objects": {"type": "array", "items": _SHORT_TEXT, "maxItems": 8},
        "actions": {"type": "array", "items": _SHORT_TEXT, "maxItems": 6},
        "setting": {"type": "string", "maxLength": 120},
    },
    "required": ["caption", "people", "objects", "actions", "setting"],
}

CAPTION_PROMPT = (
    "These frames are consecutive moments from one short video clip, in order.\n"
    "Describe what HAPPENS across them, not what a single frame shows: name the "
    "change, movement or action that unfolds.\n"
    "Be concrete and literal. Name what is visible. Do not speculate about "
    "motives, mood or backstory, and do not mention frames, images or video.\n"
    "Reply with JSON only. Keep every field short:\n"
    '  caption  one or two sentences describing the event\n'
    '  people   a few words per person, by appearance\n'
    '  objects  notable objects, one or two words each\n'
    '  actions  single verbs for what is being done\n'
    '  setting  a short phrase for where this takes place'
)


class Captioner:
    """Qwen3-VL over several frames per clip, emitting structured JSON.

    Multiple frames rather than one, so the model describes change ("walks in
    and sets a box down") instead of a still. That is the entire reason a VLM
    earns its cost here over an image captioner.

    Loaded in 4-bit NF4: Qwen ships no AWQ build of this model and the FP8 build
    needs Ada/Hopper, while this box is Ampere. bitsandbytes quantizes the bf16
    checkpoint at load time instead.
    """

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        from .config import CAPTION_MODEL

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        log.info("loading %s in 4-bit nf4", CAPTION_MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(
            CAPTION_MODEL, quantization_config=quant, device_map=device, dtype=torch.bfloat16,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
        self.tokenizer = self.processor.tokenizer
        self._prompt_len = 0
        self._prefix_fn = self._build_json_enforcer()

    def _build_json_enforcer(self):
        """Constrain decoding to the schema, so the model cannot emit non-JSON.

        Built against lm-format-enforcer's core rather than its transformers
        integration: that integration imports
        transformers.tokenization_utils.PreTrainedTokenizerBase, which
        transformers 5 moved, so it raises a misleading "transformers is not
        installed". The core enforcer needs only a vocabulary and a decoder.

        Raises if it cannot be built. Falling back to parsing free text would be
        a silent quality regression -- the whole point is that the JSON is
        guaranteed at generation time, not hoped for.
        """
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData

        tokenizer = self.tokenizer
        vocab_size = len(tokenizer)
        special = set(tokenizer.all_special_ids)

        # lm-format-enforcer needs to know which tokens begin a new word, and
        # detects that by decoding each token both alone and after a known
        # prefix: if the decoded form differs, the token carries a word break.
        #
        # This runs over the whole vocabulary (~150k tokens), so it is done with
        # two batch decodes rather than per-token calls -- the naive loop takes
        # minutes and dominates model load time.
        token_zero = tokenizer.encode("0")[-1]
        zero_len = len(tokenizer.decode([token_zero]))

        ids = [i for i in range(vocab_size) if i not in special]
        alone = tokenizer.batch_decode([[i] for i in ids])
        after = tokenizer.batch_decode([[token_zero, i] for i in ids])

        # A token starts a word when decoding it after the prefix yields MORE
        # characters than decoding it alone -- it gained a leading space. Note
        # this is a length comparison, not an inequality: plenty of tokens
        # decode differently after a prefix without starting a word, and
        # mislabelling them makes the enforcer reject valid continuations.
        regular_tokens = [
            (i, a[zero_len:], len(a[zero_len:]) > len(b))
            for i, a, b in zip(ids, after, alone)
        ]

        # Bitmask rather than a token list. With a list, every decoding step
        # materialises a Python list of up to ~150k allowed ids for every
        # sequence in the batch, which dominates generation time -- it made a
        # 15-clip video take 90s. The bitmask is a small int32 tensor that is
        # unpacked on the GPU instead.
        data = TokenEnforcerTokenizerData(
            regular_tokens=regular_tokens,
            decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            eos_token_id=tokenizer.eos_token_id,
            use_bitmask=True,
            vocab_size=vocab_size,
        )
        self._enforcer = TokenEnforcer(data, JsonSchemaParser(CAPTION_SCHEMA))
        self._vocab_size = vocab_size
        return CaptionLogitsProcessor(self)

    @staticmethod
    def pick_frames(frames: list, want: int) -> list:
        """Evenly spaced frames spanning the clip, endpoints included."""
        if len(frames) <= want:
            return frames
        step = (len(frames) - 1) / (want - 1)
        return [frames[round(i * step)] for i in range(want)]

    def describe(self, clip_frames: list[list]) -> list[dict]:
        """Caption a batch of clips. clip_frames[i] is one clip's PIL images."""
        import torch

        from .config import CAPTION_MAX_TOKENS

        messages = [
            [{"role": "user", "content": [{"type": "image"} for _ in images]
                                         + [{"type": "text", "text": CAPTION_PROMPT}]}]
            for images in clip_frames
        ]
        prompts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        inputs = self.processor(
            text=prompts, images=clip_frames, padding=True, return_tensors="pt",
        ).to(self.model.device)

        self._prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            from transformers import LogitsProcessorList

            generated = self.model.generate(
                **inputs,
                max_new_tokens=CAPTION_MAX_TOKENS,
                do_sample=False,                  # captions must be reproducible
                logits_processor=LogitsProcessorList([self._prefix_fn]),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        return [parse_caption(t) for t in texts]


class CaptionLogitsProcessor:
    """Applies the schema constraint to a batch of logits in one pass.

    transformers' prefix_allowed_tokens_fn interface wants a list of ids per
    sequence; this bypasses it and masks the score tensor directly, so no large
    Python lists are built during decoding.
    """

    def __init__(self, captioner: "Captioner"):
        self.captioner = captioner
        self._bits = None

    def __call__(self, input_ids, scores):
        import torch

        cap = self.captioner
        vocab = scores.shape[-1]

        if self._bits is None or self._bits.device != scores.device:
            self._bits = torch.arange(32, device=scores.device, dtype=torch.int32)

        mask = torch.zeros_like(scores, dtype=torch.bool)
        for row in range(input_ids.shape[0]):
            # Only the generated suffix is JSON; the prompt is not.
            generated = input_ids[row, cap._prompt_len:].tolist()
            packed = cap._enforcer.get_allowed_tokens(generated).allowed_tokens
            packed = packed.to(scores.device)
            bits = ((packed.unsqueeze(1) >> self._bits) & 1).bool().flatten()
            # The logit width can exceed the tokenizer's vocabulary: models pad
            # the embedding matrix to a round size. Those ids are not real
            # tokens, so leaving them masked off is correct.
            n = min(bits.numel(), vocab)
            mask[row, :n] = bits[:n]

        return scores.masked_fill(~mask, float("-inf"))


def parse_caption(text: str) -> dict:
    """Read the model's JSON, tolerating stray prose around it.

    Decoding is schema-constrained, so this should always be a clean parse. It
    stays tolerant anyway: one malformed caption should cost that clip its
    caption, not abort the video.
    """
    import json

    blank = {"caption": None, "people": [], "objects": [], "actions": [], "setting": None}
    if not text:
        return blank

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return blank
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return blank
    if not isinstance(data, dict):
        return blank

    def as_list(value):
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [str(value).strip()] if value else []

    caption = str(data.get("caption") or "").strip() or None
    setting = str(data.get("setting") or "").strip() or None
    return {
        "caption": caption,
        "people": as_list(data.get("people")),
        "objects": as_list(data.get("objects")),
        "actions": as_list(data.get("actions")),
        "setting": setting,
    }


def tags_text(record: dict) -> str | None:
    """Flatten the structured fields for full-text search.

    A generated tsvector column must be immutable, and jsonb_array_elements is
    set-returning, so the flattening happens here rather than in SQL.
    """
    parts = record["people"] + record["objects"] + record["actions"]
    if record.get("setting"):
        parts.append(record["setting"])
    joined = " ".join(p for p in parts if p).strip()
    return joined or None
