"""ffmpeg/ffprobe wrappers and clip segmentation."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import CLIP_LEN, FRAME_RATE, FRAME_SIZE


@dataclass
class Probe:
    duration_s: float
    fps: float
    width: int
    height: int
    # Whether there is anything to transcribe. Whisper does not fail politely
    # on a file with no audio track -- it crashes inside its demuxer with
    # "tuple index out of range" -- so the caller has to know in advance.
    # Silent video is common: screen recordings, GIF conversions, drone footage.
    has_audio: bool = False


def probe(path: Path) -> Probe:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    all_streams = data.get("streams") or []
    streams = [x for x in all_streams if x.get("codec_type") == "video"]
    if not streams:
        raise ValueError(f"{path.name}: no video stream")
    has_audio = any(x.get("codec_type") == "audio" for x in all_streams)
    s = streams[0]

    duration = float(data.get("format", {}).get("duration") or s.get("duration") or 0)
    if duration <= 0:
        raise ValueError(f"{path.name}: no usable duration")

    num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0

    return Probe(duration, fps, int(s.get("width", 0)), int(s.get("height", 0)),
                 has_audio=has_audio)


@dataclass
class Frame:
    idx: int
    t: float
    path: Path


def extract_frames(src: Path, dest: Path, rate: float = FRAME_RATE,
                   size: int = FRAME_SIZE) -> list[Frame]:
    """Sample the video and write pre-scaled JPEGs into dest.

    Frames are scaled here because SigLIP resizes to a square without
    preserving aspect ratio anyway, so doing it in ffmpeg is equivalent and
    much cheaper than decoding full-resolution stills later.

    The fps filter emits frames at t = 0, 1/rate, 2/rate ... so a frame's index
    gives its timestamp arithmetically; no per-frame probing needed.
    """
    dest.mkdir(parents=True, exist_ok=True)
    base = ["ffmpeg", "-v", "error", "-i", str(src),
            "-vf", f"fps={rate:g},scale={size}:{size}:flags=bicubic", "-q:v", "3"]
    out = str(dest / "f_%05d.jpg")

    # -fps_mode arrived in ffmpeg 5.0; older builds -- Kaggle's image among them
    # -- want the -vsync spelling and reject the new one outright. Try the
    # current flag, fall back rather than requiring a particular ffmpeg.
    attempts = [base + ["-fps_mode", "passthrough", out],
                base + ["-vsync", "passthrough", out]]
    last = None
    for cmd in attempts:
        done = subprocess.run(cmd, capture_output=True)
        if done.returncode == 0:
            break
        last = done
    else:
        # capture_output hides ffmpeg's reason, and "exit status 1" is not a
        # diagnosis -- this failed on Kaggle once and said nothing useful.
        raise RuntimeError(
            f"ffmpeg could not extract frames from {src.name}: "
            f"{(last.stderr or b'').decode(errors='replace').strip()[:400]}")
    files = sorted(dest.glob("f_*.jpg"))
    if not files:
        raise ValueError(f"{src.name}: ffmpeg produced no frames")
    return [Frame(i, i / rate, p) for i, p in enumerate(files)]


@dataclass
class Clip:
    idx: int
    start_s: float
    end_s: float
    frames: list[Frame] = field(default_factory=list)


def segment(duration: float, frames: list[Frame], clip_len: float = CLIP_LEN) -> list[Clip]:
    """Cut the video into a fixed grid and assign each frame to its clip.

    A trailing remainder shorter than half a clip is folded into the previous
    clip, so a 150.02s video yields 15 clips rather than 15 plus a 0.02s stub.
    """
    if duration <= 0 or clip_len <= 0:
        return []

    bounds = []
    start = 0.0
    while start < duration:
        bounds.append((start, min(start + clip_len, duration)))
        start += clip_len
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < clip_len / 2:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()

    clips = [Clip(i, lo, hi) for i, (lo, hi) in enumerate(bounds)]
    for f in frames:
        clips[min(int(f.t / clip_len), len(clips) - 1)].frames.append(f)
    return clips
