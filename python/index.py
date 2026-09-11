#!/usr/bin/env python3
"""Index one video, end to end.

Everything that turns a video file into searchable data happens here: ffmpeg,
SigLIP, and later Whisper and the captioner. Go starts this script, reads the
JSON lines it prints, and writes the results to the database.

Nothing in here touches Postgres. That keeps the free-tier deployment simple --
if this file is absent, the server has no indexing path at all.

Output is one JSON object per line on stdout:

    {"event": "progress", "stage": "frames", "done": 0, "total": 1}
    {"event": "result",   "video": {...}, "clips": [...]}
    {"event": "error",    "message": "..."}

Stages run serially. They are independent enough to overlap later, but serial
is easier to read and the GPU is the bottleneck either way.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.lib import video as vid
from python.lib.config import (CAPTION_MODEL, CLIP_LEN, FRAME_RATE,
                               PIPELINE_VERSION, SCRATCH, device_for)


# The JSON-lines protocol owns stdout, and it is fragile: anything else printed
# there lands in the middle of a line and Go drops the whole thing. yt-dlp does
# exactly that -- it writes a \r progress bar with no newline, which swallowed
# the download and probe events until this was fixed.
#
# So the real stream is captured once, at import, and `emit` writes to it
# directly. main() then points sys.stdout at stderr for the whole run, which
# makes a stray print from any library harmless instead of protocol-breaking.
_PROTOCOL = sys.stdout


def emit(**payload) -> None:
    """One JSON object per line, flushed, so Go sees progress as it happens."""
    print(json.dumps(payload), file=_PROTOCOL, flush=True)


def _emit_progress(stage: str, done: int = 0, total: int = 1) -> None:
    emit(event="progress", stage=stage, done=done, total=total)


def b64(arr) -> str:
    import numpy as np
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32)).decode("ascii")


def index_video(source: Path, video_id: str, *, models=None, keep_frames: bool = False,
                on_progress=None) -> dict:
    """Index one file and return the payload Go will persist.

    Pass an already-loaded IndexModels when indexing many videos in one
    process -- loading the models costs far more than running them on a single
    video.

    on_progress defaults to printing JSON lines for Go to read. Callers that
    are not Go (the bulk script) pass their own, so progress does not collide
    with their output.
    """
    import numpy as np

    progress = on_progress if on_progress is not None else _emit_progress
    started = time.monotonic()

    progress("probe")
    meta = vid.probe(source)

    # The directory is keyed by process as well as video: two indexers running
    # the same video would otherwise share a scratch directory, and the first to
    # finish would delete the frames the second is still reading.
    work = SCRATCH / f"{video_id}.{os.getpid()}"
    try:
        progress("frames")
        frames = vid.extract_frames(source, work)
        clips = vid.segment(meta.duration_s, frames)
        if not clips:
            raise ValueError(f"{video_id}: no clips from a {meta.duration_s:.1f}s video")

        progress("embed", 0, len(frames))
        if models is None:
            from python.lib.indexing import IndexModels
            models = IndexModels(device_for("index"))

        from PIL import Image
        images = [Image.open(f.path).convert("RGB") for f in frames]
        vectors = models.siglip.embed_images(images)
        for im in images:
            im.close()
        progress("embed", len(frames), len(frames))

        if len(vectors) != len(frames):
            raise ValueError(f"{video_id}: {len(vectors)} vectors for {len(frames)} frames")
        by_idx = {f.idx: vectors[i] for i, f in enumerate(frames)}

        # --- speech --------------------------------------------------------
        # Reads the source file directly, so it does not depend on the frames.
        #
        # Silent video is skipped rather than attempted. faster-whisper does not
        # fail politely on a file with no audio track: it crashes inside its
        # demuxer with "tuple index out of range", which would take down the
        # whole index run over a stage that had nothing to do. Screen
        # recordings, GIF conversions and drone footage all arrive this way.
        transcript, words, language = [], [], None
        if meta.has_audio:
            progress("transcribe")
            transcript, words, language = models.whisper.transcribe(source)
        else:
            progress("transcribe", 0, 0)

        from python.lib.indexing import speech_for_clips
        clip_speech = speech_for_clips(clips, words)

        spoken = [i for i, text in enumerate(clip_speech) if text]
        speech_vectors = {}
        if spoken:
            progress("embed_speech", 0, len(spoken))
            rows = models.bge.embed_texts([clip_speech[i] for i in spoken])
            speech_vectors = {i: rows[j] for j, i in enumerate(spoken)}
            progress("embed_speech", len(spoken), len(spoken))

        # --- captions ------------------------------------------------------
        # Several frames per clip, not one, so the model describes change
        # rather than a still. That is the whole reason a VLM earns its cost.
        from python.lib.indexing import Captioner, tags_text
        from python.lib.config import CAPTION_BATCH, CAPTION_FRAMES

        progress("caption", 0, len(clips))
        captions: list[dict] = []
        for start in range(0, len(clips), CAPTION_BATCH):
            chunk = clips[start:start + CAPTION_BATCH]
            batch_images = []
            for clip in chunk:
                picked = Captioner.pick_frames(clip.frames, CAPTION_FRAMES)
                batch_images.append([Image.open(f.path).convert("RGB") for f in picked])

            captions.extend(models.captioner.describe(batch_images))

            for group in batch_images:
                for im in group:
                    im.close()
            progress("caption", min(start + CAPTION_BATCH, len(clips)), len(clips))

        described = [i for i, c in enumerate(captions) if c["caption"]]
        caption_vectors = {}
        if described:
            progress("embed_caption", 0, len(described))
            # Captions are embedded with bge-m3, not SigLIP: SigLIP is weak at
            # text-to-text, and a caption is matched against query text.
            texts = [
                " ".join(filter(None, [captions[i]["caption"], tags_text(captions[i])]))
                for i in described
            ]
            rows = models.bge.embed_texts(texts)
            caption_vectors = {i: rows[j] for j, i in enumerate(described)}
            progress("embed_caption", len(described), len(described))

        out_clips = []
        previous = None
        for clip in clips:
            rows = [by_idx[f.idx] for f in clip.frames if f.idx in by_idx]
            record = {
                "idx": clip.idx,
                "start_s": clip.start_s,
                "end_s": clip.end_s,
                "visual": None,
                "static_score": None,
                "novelty": None,
                "frames": [],
                "caption": captions[clip.idx]["caption"],
                "people": captions[clip.idx]["people"],
                "objects": captions[clip.idx]["objects"],
                "actions": captions[clip.idx]["actions"],
                "setting": captions[clip.idx]["setting"],
                "tags_text": tags_text(captions[clip.idx]),
                "caption_vec": (b64(caption_vectors[clip.idx])
                                if clip.idx in caption_vectors else None),
                "speech": clip_speech[clip.idx] or None,
                "speech_vec": (b64(speech_vectors[clip.idx])
                               if clip.idx in speech_vectors else None),
                "lang": language if clip_speech[clip.idx] else None,
            }
            if rows:
                stacked = np.stack(rows)
                mean = stacked.mean(axis=0)
                norm = float(np.linalg.norm(mean))
                mean = mean / norm if norm > 1e-12 else mean

                record["visual"] = b64(mean)
                # Frames barely differing from the clip mean mean a static
                # clip: black frames, title cards, slideshows. Recorded, not
                # acted on.
                record["static_score"] = float((stacked @ mean).mean())
                if previous is not None:
                    record["novelty"] = float(1.0 - float(previous @ mean))
                previous = mean
                record["frames"] = [
                    {"idx": f.idx, "t_s": f.t, "embedding": b64(by_idx[f.idx])}
                    for f in clip.frames if f.idx in by_idx
                ]
            out_clips.append(record)

        return {
            "video": {
                "id": video_id,
                "duration_s": meta.duration_s,
                "fps": meta.fps,
                "width": meta.width,
                "height": meta.height,
            },
            "clips": out_clips,
            "transcript": transcript,
            "language": language,
            "pipeline": PIPELINE_VERSION,
            "caption_model": CAPTION_MODEL if described else None,
            "stats": {
                "clips": len(out_clips),
                "frames": len(frames),
                "transcript_segments": len(transcript),
                "clips_with_speech": len(spoken),
                "clips_with_caption": len(described),
                "clip_len_s": CLIP_LEN,
                "frame_rate": FRAME_RATE,
                "took_s": round(time.monotonic() - started, 2),
            },
        }
    finally:
        if not keep_frames:
            shutil.rmtree(work, ignore_errors=True)


def index_url(url: str, video_id: str, *, models=None, keep_frames: bool = False,
              on_progress=None) -> dict:
    """Fetch a link, index it, and delete the bytes.

    The whole point of doing this here rather than in Go: the download, the
    locator and the clips are one decision. Go sends a URL and gets back
    something it can write to the database, and never handles a video file.

    The file is removed in a finally block, so a failure mid-index does not
    leave a download behind. What persists is the locator.
    """
    import shutil

    from python.lib import fetch as fetcher

    progress = on_progress if on_progress is not None else _emit_progress
    progress("resolve")
    got = fetcher.fetch(url, video_id, on_progress=progress)
    try:
        payload = index_video(got.path, video_id, models=models,
                              keep_frames=keep_frames, on_progress=progress)
    finally:
        shutil.rmtree(got.path.parent, ignore_errors=True)

    # The locator and title are the two things only the fetch step knows, and
    # the only things in the payload that describe the video rather than its
    # contents.
    payload["video"]["locator"] = got.locator
    payload["video"]["title"] = got.title
    return payload


def main() -> int:
    # Nothing but emit() may write to the protocol stream; see _PROTOCOL.
    with contextlib.redirect_stdout(sys.stderr):
        return _main()


def _main() -> int:
    ap = argparse.ArgumentParser(description="Index one video into searchable data.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="link to fetch with yt-dlp, then index")
    src.add_argument("--source", help="path to a video file already on disk")
    ap.add_argument("--video-id", required=True, help="stable id, also the database key")
    ap.add_argument("--keep-frames", action="store_true", help="leave extracted frames on disk")
    args = ap.parse_args()

    try:
        if args.url:
            payload = index_url(args.url, args.video_id, keep_frames=args.keep_frames)
        else:
            payload = index_video(Path(args.source), args.video_id, keep_frames=args.keep_frames)
    except Exception as exc:
        emit(event="error", message=f"{type(exc).__name__}: {exc}")
        return 1

    emit(event="result", **payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
