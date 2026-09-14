"""Turn a submitted link into a local file plus the locator that outlives it.

This is the indexing path's front door. Go hands over a URL and never touches
the bytes: yt-dlp resolves the link, downloads a stream, and the file is deleted
once the clips are made. What survives is the locator -- a pointer at wherever
the video already lives -- because this project never stores video bytes.

Two things come out of one yt-dlp probe:

    locator   where playback points forever (youtube id, or the original URL)
    source    a temporary file on disk, valid for the length of one index run

Import-safe: yt_dlp is imported inside the functions, so `python/lib/config.py`
and the search path never pull it in.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from python.lib.config import MAX_SOURCE_DURATION, SCRATCH

# stderr: index.py's stdout carries the JSON protocol and must stay clean.
log = logging.getLogger(__name__)

# How long an abandoned download may sit before the next fetch clears it. Long
# enough that a download still running is never mistaken for a dead one.
STALE_DOWNLOAD_AGE = 6 * 60 * 60

# yt-dlp will happily read `file://`, and a link is user input on an endpoint
# that may be reachable from a browser. Only these two schemes get through.
ALLOWED_SCHEMES = ("http", "https")

# Cap the stream so an accidental 4K submission does not spend an hour of GPU
# time. 1080p is already far above what SigLIP sees at 384px.
# 720p: frames are downscaled to 384px anyway, so higher resolution costs
# download time and disk for no retrieval gain.
FORMAT = ("bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
          "best[height<=720][ext=mp4]/best")

# YouTube serves different player APIs per client and refuses some as bot
# traffic depending on the address. No single client works everywhere, so each
# is tried in turn, most reliable first.
CLIENT_STRATEGIES: list[tuple[str, dict]] = [
    ("ios", {"player_client": ["ios"]}),
    ("web_safari", {"player_client": ["web_safari"]}),
    ("tv", {"player_client": ["tv"]}),
    ("mweb", {"player_client": ["mweb"]}),
    ("android_embedded", {"player_client": ["android_embedded", "web"]}),
    ("default", {}),
]


def _with_client(options: dict, extractor: dict) -> dict:
    """Copy options, pointing the YouTube extractor at one player client."""
    out = dict(options)
    if extractor:
        args = dict(out.get("extractor_args") or {})
        args["youtube"] = {**args.get("youtube", {}), **extractor}
        out["extractor_args"] = args
    return out


class FetchError(RuntimeError):
    """A link that cannot be indexed, with a reason meant for a person."""


class _Logger:
    """Route yt-dlp's own output to stderr.

    index.py's stdout carries a JSON-lines protocol. yt-dlp writes to stdout by
    default, and its progress bar uses \r with no newline, so its text ends up
    prepended to a JSON object and the line is dropped. Errors are still worth
    seeing, so they go to the log stream rather than nowhere.
    """

    @staticmethod
    def debug(msg: str) -> None: pass

    @staticmethod
    def info(msg: str) -> None: pass

    @staticmethod
    def warning(msg: str) -> None:
        print(f"yt-dlp: {msg}", file=sys.stderr)

    @staticmethod
    def error(msg: str) -> None:
        print(f"yt-dlp: {msg}", file=sys.stderr)


# Shared by probe and fetch, so the two cannot drift apart on the settings that
# matter: no playlists, nothing on stdout.
# A Netscape cookies.txt, if one is available. The client fallback above
# handles most refusals; a bot challenge needs cookies.
COOKIES = os.environ.get("YTDLP_COOKIES", "")


def _options(**extra) -> dict:
    options = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,      # our own hook reports progress, in JSON
        "logger": _Logger,
        "noplaylist": True,      # a playlist link means its first video, not 200
        **extra,
    }
    if COOKIES and Path(COOKIES).is_file():
        options["cookiefile"] = COOKIES
    return options


@dataclass(slots=True)
class Fetched:
    path: Path          # the downloaded file, caller deletes it
    locator: dict       # what goes in videos.locator, forever
    title: str
    duration_s: float


def check_url(url: str) -> str:
    """Reject anything that is not a plain web link, before yt-dlp sees it."""
    url = (url or "").strip()
    if not url:
        raise FetchError("no url given")
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise FetchError("url must be http or https")
    if not parsed.netloc:
        raise FetchError("url has no host")
    return url


def locator_for(info: dict, submitted: str) -> dict:
    """Where this video plays back from, once the local file is gone.

    YouTube gets its own kind so the frontend can embed the player and seek,
    and so nothing depends on a URL format Google is free to change. Everything
    else keeps the URL the user submitted rather than the direct stream URL
    yt-dlp resolved: those are signed and expire within hours, which is exactly
    the property a stored locator must not have.
    """
    extractor = (info.get("extractor_key") or info.get("extractor") or "").lower()
    if extractor.startswith("youtube") and info.get("id"):
        # offset 0: a submitted video is indexed whole, unlike the QVHighlights
        # excerpts, which carry their position in the original.
        return {"kind": "youtube", "id": info["id"], "offset": 0}
    return {"kind": "http", "url": submitted}


def probe(url: str) -> dict:
    """Resolve a link without downloading it."""
    import yt_dlp

    url = check_url(url)
    base = _options(skip_download=True)

    info, last = None, None
    for name, extractor in CLIENT_STRATEGIES:
        try:
            with yt_dlp.YoutubeDL(_with_client(base, extractor)) as ydl:
                info = ydl.extract_info(url, download=False)
            log.debug("probe succeeded with client strategy %s", name)
            break
        except Exception as exc:
            log.info("probe: client %s refused (%s)", name, _reason(exc))
            last = exc
    if info is None:
        raise FetchError(f"could not read that link: {_reason(last)}") from last

    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise FetchError("that link has no video in it")
        info = entries[0]
    return info


def fetch(url: str, video_id: str, on_progress=None) -> Fetched:
    """Download one video and return it with the locator that outlives it."""
    import yt_dlp

    url = check_url(url)
    info = probe(url)

    duration = float(info.get("duration") or 0.0)
    if MAX_SOURCE_DURATION and duration > MAX_SOURCE_DURATION:
        raise FetchError(
            f"that video is {duration / 60:.0f} minutes; the limit is "
            f"{MAX_SOURCE_DURATION / 60:.0f}. Indexing is about a minute of GPU "
            f"time per minute of video")

    downloads = SCRATCH / "downloads"
    sweep_stale(downloads)

    # Its own directory, keyed by a random suffix: two people adding the same
    # link at the same time must not write to one path.
    work = downloads / f"{video_id}.{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=True)

    def hook(status: dict) -> None:
        if on_progress is None or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        got = status.get("downloaded_bytes") or 0
        # Reported as a percentage rather than bytes: every other stage counts
        # items, and "download 41/100" reads better than "download 8.4e6/2.0e7".
        on_progress("download", int(100 * got / total) if total else 0, 100)

    base = _options(
        format=FORMAT,
        outtmpl=str(work / "source.%(ext)s"),
        merge_output_format="mp4",
        progress_hooks=[hook],
        retries=3,
        concurrent_fragment_downloads=4,
    )

    # A refusal is per-client, so it is worth retrying with another player.
    # Partial files are cleared first, or yt-dlp resumes a truncated download
    # from the previous client's format.
    last = None
    for name, extractor in CLIENT_STRATEGIES:
        try:
            with yt_dlp.YoutubeDL(_with_client(base, extractor)) as ydl:
                ydl.download([url])
        except Exception as exc:
            log.info("download: client %s refused (%s)", name, _reason(exc))
            last = exc
            for stale in work.iterdir():
                stale.unlink(missing_ok=True)
            continue
        if any(p.is_file() for p in work.iterdir()):
            log.info("download succeeded with client strategy %s", name)
            break
        last = last or FetchError("download produced no file")

    files = sorted(p for p in work.iterdir() if p.is_file())
    if not files:
        shutil.rmtree(work, ignore_errors=True)
        raise FetchError(f"download failed: {_reason(last)}" if last
                         else "download produced no file")

    return Fetched(
        path=max(files, key=lambda p: p.stat().st_size),
        locator=locator_for(info, url),
        title=(info.get("title") or "").strip(),
        duration_s=duration,
    )


def sweep_stale(downloads: Path, older_than: float = STALE_DOWNLOAD_AGE) -> int:
    """Delete downloads left behind by a run that was killed.

    index.py removes its own download in a finally block, but a finally does not
    run when the process is killed -- and it is killed whenever the browser
    disconnects, because Go ties the subprocess to the request context. So
    cancelling an add-video halfway leaves a video file on disk, and the project
    stores no video bytes by design.

    Sweeping on the way in rather than on a timer keeps it to one place, and the
    age cut-off means a concurrent download in progress is never touched.
    """
    if not downloads.is_dir():
        return 0
    cutoff = time.time() - older_than
    removed = 0
    for entry in downloads.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except OSError:
            continue          # a racing sweep got there first; not our problem
    return removed


def _reason(exc: Exception) -> str:
    """yt-dlp errors carry an ANSI-coloured prefix meant for a terminal."""
    text = str(exc).replace("\x1b[0;31mERROR:\x1b[0m", "").strip()
    text = text.replace("ERROR:", "").strip()
    return text.splitlines()[0][:300] if text else type(exc).__name__
