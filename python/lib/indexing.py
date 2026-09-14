"""Models used only when indexing.

Kept apart from lib/models.py so the search path's import graph never reaches
Whisper or the captioner. A machine serving search should not need faster-whisper
installed at all, and scripts/check-boundary.sh enforces that.
"""
from __future__ import annotations

import logging
import re
import time

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

    def release(self, *names: str) -> None:
        """Drop models from VRAM once their stage is done.

        Stages run in order, so a finished model is only occupying memory the
        next stage could use -- which decides how large a caption batch fits.
        Reloading is a property away, so releasing something still needed
        costs a load rather than an error. Note caption embedding runs *after*
        captioning, so bge is not finished when the captioner starts.
        """
        import gc

        for name in names:
            attr = f"_{name}"
            if getattr(self, attr, None) is None:
                continue
            setattr(self, attr, None)
            log.info("released %s from %s", name, self.device)

        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            # Python dropping the reference is not enough: torch keeps the
            # blocks in its caching allocator until told otherwise.
            torch.cuda.empty_cache()


# Lengths belong in the schema, not just the prompt: constrained decoding
# blocks EOS until the object is complete, so an unbounded field means the
# model elaborates until the token limit and the JSON never closes.
_SHORT_TEXT = {"type": "string", "maxLength": 90}

# What actually bounds caption length: the model writes until something stops
# it, and a longer allowance buys trailing scenery rather than more event.
# Kept tight; tidy_caption repairs the ragged edge it leaves.
CAPTION_MAX_CHARS = 260

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string", "maxLength": CAPTION_MAX_CHARS},
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
    # Without this the model narrates frame by frame, which runs long and
    # gets cut. Asking for a full stop as well backfires: it splits into more
    # sentences of scenery rather than fewer.
    "Summarise the event as a whole. Do not walk through the frames one by one "
    "and do not chain clauses with \"then\" -- say what happened, once.\n"
    "Reply with JSON only. Keep every field short:\n"
    '  caption  one or two sentences describing the event\n'
    '  people   a few words per person, by appearance\n'
    '  objects  notable objects, one or two words each\n'
    '  actions  single verbs for what is being done\n'
    '  setting  a short phrase for where this takes place'
)


def _has_quantization_config(model_id: str) -> bool:
    """Whether a checkpoint carries its own quantization settings.

    Reads config.json directly rather than loading the model, because this
    decides how to load it. A repo id that is not a local directory is treated
    as full precision, which is what the published Qwen weights are.
    """
    import json
    from pathlib import Path as _Path

    config = _Path(model_id) / "config.json"
    if not config.is_file():
        return False
    try:
        return "quantization_config" in json.loads(config.read_text())
    except Exception:
        return False


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

        # bfloat16 only where the card supports it. Older GPUs emulate it
        # rather than failing, which is far slower than float16.
        want_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
        compute = torch.bfloat16 if want_bf16 else torch.float16
        log.info("caption compute dtype: %s", compute)

        # A pre-quantized checkpoint carries its own settings, and passing
        # BitsAndBytesConfig alongside them conflicts -- so only quantize what
        # is still full precision.
        if _has_quantization_config(CAPTION_MODEL):
            log.info("loading %s (already quantized)", CAPTION_MODEL)
            self.model = AutoModelForImageTextToText.from_pretrained(
                CAPTION_MODEL, device_map=device,
            ).eval()
        else:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute,
                bnb_4bit_use_double_quant=True,
            )
            log.info("loading %s in 4-bit nf4", CAPTION_MODEL)
            self.model = AutoModelForImageTextToText.from_pretrained(
                CAPTION_MODEL, quantization_config=quant, device_map=device, dtype=compute,
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
        """Caption a batch of clips. clip_frames[i] is one clip's PIL images.

        Generates unconstrained first, which is several times faster: the
        schema enforcer runs a Python parser over the whole vocabulary once per
        token per sequence and dominates the stage. The model usually emits
        valid JSON anyway; any clip whose output does not parse is generated
        again with the enforcer, which cannot produce malformed output.
        """
        results = self._generate(clip_frames, constrain=False)

        retry = [i for i, r in enumerate(results) if not r["caption"]]
        if retry:
            log.info("caption: %d/%d clips need the schema enforcer",
                     len(retry), len(clip_frames))
            fixed = self._generate([clip_frames[i] for i in retry], constrain=True)
            for i, r in zip(retry, fixed):
                results[i] = r
        return results

    def _generate(self, clip_frames: list[list], constrain: bool) -> list[dict]:
        """One generate() over a batch, with or without the schema constraint."""
        import torch

        from .config import CAPTION_MAX_TOKENS

        if not clip_frames:
            return []

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
        extra = {}
        if constrain:
            from transformers import LogitsProcessorList

            if hasattr(self._prefix_fn, "reset"):
                self._prefix_fn.reset()
            extra["logits_processor"] = LogitsProcessorList([self._prefix_fn])

        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=CAPTION_MAX_TOKENS,
                do_sample=False,                  # captions must be reproducible
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **extra,
            )

        elapsed = time.perf_counter() - started
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        tokens = int(trimmed.shape[0] * trimmed.shape[1])

        # Kept because it is what found the enforcer in the first place: the
        # split between masking and everything else is the difference between
        # "the GPU is slow" and "the CPU is the bottleneck".
        fn = self._prefix_fn
        if constrain and getattr(fn, "calls", 0):
            log.info(
                "caption batch (constrained): %d clips, %d steps, %d tokens in "
                "%.1fs (%.1f tok/s) | mask %.1fs (%.0f%%), of which enforcer "
                "%.1fs (%.0f%%) | rest %.1fs",
                len(clip_frames), fn.calls, tokens, elapsed,
                tokens / elapsed if elapsed else 0,
                fn.seconds, 100 * fn.seconds / elapsed if elapsed else 0,
                fn.enforcer_seconds,
                100 * fn.enforcer_seconds / elapsed if elapsed else 0,
                elapsed - fn.seconds)
        else:
            log.info("caption batch: %d clips, %d tokens in %.1fs (%.1f tok/s)",
                     len(clip_frames), tokens, elapsed,
                     tokens / elapsed if elapsed else 0)

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
        # Cheap counters, always on: two ints and a float per decode step cost
        # nothing against a forward pass, and without them "captioning is slow"
        # is a feeling rather than a measurement.
        self.calls = 0
        self.rows = 0
        self.seconds = 0.0
        self.enforcer_seconds = 0.0

    def reset(self) -> None:
        self.calls = self.rows = 0
        self.seconds = self.enforcer_seconds = 0.0

    def __call__(self, input_ids, scores):
        import torch

        started = time.perf_counter()
        self.calls += 1
        self.rows += input_ids.shape[0]

        cap = self.captioner
        vocab = scores.shape[-1]

        if self._bits is None or self._bits.device != scores.device:
            self._bits = torch.arange(32, device=scores.device, dtype=torch.int32)

        mask = torch.zeros_like(scores, dtype=torch.bool)
        for row in range(input_ids.shape[0]):
            # Only the generated suffix is JSON; the prompt is not.
            generated = input_ids[row, cap._prompt_len:].tolist()
            # Timed separately: this is the pure-Python part, and whether it
            # dominates decides whether the fix is the enforcer or the GPU.
            _t = time.perf_counter()
            packed = cap._enforcer.get_allowed_tokens(generated).allowed_tokens
            self.enforcer_seconds += time.perf_counter() - _t
            packed = packed.to(scores.device)
            bits = ((packed.unsqueeze(1) >> self._bits) & 1).bool().flatten()
            # The logit width can exceed the tokenizer's vocabulary: models pad
            # the embedding matrix to a round size. Those ids are not real
            # tokens, so leaving them masked off is correct.
            n = min(bits.numel(), vocab)
            mask[row, :n] = bits[:n]

        self.seconds += time.perf_counter() - started

        return scores.masked_fill(~mask, float("-inf"))


# Words that leave a caption hanging when the text is cut after them: dropping
# a severed word off "...into a glass and" leaves the "and" stranded.
_DANGLING = frozenset("""
a an the and or but then while as with of to in on at by for from into onto
before after that which who whose near over under behind toward towards
is are was were be been being has have had its his her their this these those
""".split())

_SEPARATORS = " ,;:-\u2013\u2014"


def tidy_caption(text: str | None) -> str | None:
    """Close off a caption that the decoder cut short.

    The length ceiling ends the string wherever it lands, mid-word if need be.
    Most of the content is still there, so this drops the severed tail rather
    than trimming back to the last full stop. Captions under the ceiling only
    need the closing punctuation the model tends to omit.
    """
    text = (text or "").strip()
    if not text:
        return None
    if text[-1] in ".!?":
        return text

    # Only a caption at the ceiling was cut; anything shorter ended by choice.
    cut = len(text) >= CAPTION_MAX_CHARS - 2
    if cut and text[-1] not in _SEPARATORS:
        # Ending on a separator proves the last word finished before the cut.
        # Ending on a letter proves nothing, and half a word ("...the animals
        # are now visible, ccc") is worse in the index than a missing one.
        head = text.rpartition(" ")[0]
        text = head or text

    words = text.rstrip(_SEPARATORS).split()
    # Never strip a caption down to a stub chasing a tidy ending -- past a few
    # words the dangling tail costs less than the content would.
    while len(words) > 3 and words[-1].lower().strip(_SEPARATORS) in _DANGLING:
        words.pop()
    text = " ".join(words).rstrip(_SEPARATORS)
    if not text:
        return None
    return _drop_orphan(text + ".") if cut else text + "."


_SENTENCE_BREAK = re.compile(r"(?<=[.!?]) +")


def _drop_orphan(text: str) -> str:
    """Discard a final sentence the cut reduced to a stub.

    When the ceiling lands just after the model starts a new sentence, closing
    it yields an orphan -- "...about football. The time stamp." Those few words
    carry no event and the embedding is better without them. Only a short tail
    goes: a long final clause is content, however abruptly it ends.
    """
    parts = _SENTENCE_BREAK.split(text)
    if len(parts) > 1 and len(parts[-1].split()) < 4:
        return " ".join(parts[:-1])
    return text


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

    caption = tidy_caption(str(data.get("caption") or ""))
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
