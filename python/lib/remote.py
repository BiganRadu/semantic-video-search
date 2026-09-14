"""Run a job on Kaggle's free GPU and bring the answer back.

For deployments with no RAM for the models. Go is unchanged: it still runs
search.py and index.py and speaks the same JSON protocol; only what happens
behind that protocol moves.

Kaggle offers no way to hold a process open and call into it, so a job is:
push a notebook, poll until it finishes, read its output. That costs minutes
per job, which is inherent to the approach rather than a fault in it.

The notebook carries its own code as an embedded tarball, so the kernel always
runs what is in this repo. Weights are the exception -- they come from attached
datasets, being large and unchanging. See models/README.md.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)

# The kernel marks its answer so it can be found in a log that also holds
# every warning and progress bar the notebook produced.
RESULT_MARKER = "__VS_RESULT__"

KAGGLE_USER = os.environ.get("KAGGLE_USERNAME", "biganradu335ca")

# One slug per job kind, reused: pushing again is a new version rather than a
# new notebook, so the account does not accumulate one kernel per search.
SLUGS = {"search": "vs-search-worker", "index": "vs-index-worker"}

# Attached weights, per job kind. Search omits the captioner rather than
# mounting gigabytes it never opens.
# vs-config carries the database URL. A dataset rather than a Kaggle Secret
# because secrets are not preserved when a notebook is pushed through the API,
# while dataset attachments are.
CONFIG_DATASET = f"{KAGGLE_USER}/vs-config"

DATASETS = {
    "search": [CONFIG_DATASET, f"{KAGGLE_USER}/vs-siglip2-so400m", f"{KAGGLE_USER}/vs-bge-m3"],
    "index": [
        CONFIG_DATASET,
        f"{KAGGLE_USER}/vs-siglip2-so400m",
        f"{KAGGLE_USER}/vs-bge-m3",
        # Pre-quantized: ~3 GB instead of 8.3, and no quantize pass at load.
        f"{KAGGLE_USER}/qwen3-vl-4b-4bit",
        f"{KAGGLE_USER}/vs-faster-whisper-turbo",
    ],
}

# How long to wait before giving up. Search reaching its limit means something
# is wrong rather than merely slow.
TIMEOUTS = {"search": 15 * 60, "index": 3 * 60 * 60}

POLL_SECONDS = float(os.environ.get("KAGGLE_POLL_SECONDS", 10))

# What each kind needs that Kaggle's image lacks. Kept short: every entry is
# seconds added to every job, and torch and transformers are already there.
PIPS = {
    "search": ["psycopg[binary]", "sentencepiece", "protobuf"],
    # bitsandbytes loads the captioner in 4-bit and is not in Kaggle's image.
    "index": ["psycopg[binary]", "sentencepiece", "protobuf", "bitsandbytes",
              "yt-dlp", "faster-whisper", "lm-format-enforcer"],
}


class RemoteError(RuntimeError):
    """A job that did not come back, with a reason meant for a person."""


@dataclass
class Job:
    kind: str
    payload: dict
    gpu: bool = True
    datasets: list[str] = field(default_factory=list)


def _api():
    """An authenticated Kaggle client.

    Imported lazily: the package authenticates on import and raises without
    credentials, which should not happen just because this module was imported.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _bundle() -> str:
    """python/ as a base64 tarball, for the notebook to unpack.

    Source only: the bundle is uploaded with every job, so its size is paid
    repeatedly. The whole tree compresses to tens of kilobytes.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted((ROOT / "python").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tar.add(path, arcname=str(path.relative_to(ROOT)))
    return base64.b64encode(buf.getvalue()).decode()


def _notebook(job: Job) -> dict:
    """The notebook that runs one job.

    One cell, so the job either raises or prints a result -- a multi-cell
    notebook that fails partway still reports as complete.
    """
    body = _KERNEL_SOURCE % {
        "bundle": _bundle(),
        "payload": base64.b64encode(json.dumps(job.payload).encode()).decode(),
        "kind": job.kind,
        "marker": RESULT_MARKER,
        "pips": repr(PIPS.get(job.kind, [])),
    }
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": body.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"language": "python", "name": "python3",
                                    "display_name": "Python 3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def _metadata(job: Job, slug: str) -> dict:
    return {
        "id": f"{KAGGLE_USER}/{slug}",
        "title": slug.replace("-", " "),
        "code_file": "kernel.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": job.gpu,
        # Ask for a T4: Kaggle's torch has no kernels for the older P100, and
        # without this which GPU a job gets is luck.
        **({"machine_shape": "NvidiaTeslaT4"} if job.gpu else {}),
        "enable_internet": True,
        "dataset_sources": job.datasets or DATASETS.get(job.kind, []),
        "competition_sources": [],
        "kernel_sources": [],
    }


def run(job: Job, on_progress=None, timeout: float | None = None) -> dict:
    """Push the job, wait for it, and return what it printed.

    on_progress(stage, done, total) is called as the kernel moves through
    Kaggle's own states, so the caller can show something during the minutes
    this takes. It is not pipeline progress -- the kernel's stdout is not
    readable until it finishes.
    """
    slug = SLUGS[job.kind]
    timeout = timeout or TIMEOUTS[job.kind]
    api = _api()

    def say(stage, done=0, total=0):
        if on_progress:
            on_progress(stage, done, total)

    work = Path(tempfile.mkdtemp(prefix="vs-kaggle-"))
    try:
        (work / "kernel.ipynb").write_text(json.dumps(_notebook(job)))
        (work / "kernel-metadata.json").write_text(json.dumps(_metadata(job, slug)))

        say("submitting")
        started = time.monotonic()
        api.kernels_push(str(work))
        log.info("pushed %s as %s/%s", job.kind, KAGGLE_USER, slug)

        ref = f"{KAGGLE_USER}/{slug}"
        state = _wait(api, ref, started, timeout, say)
        say("collecting")
        return _collect(api, ref, work, state)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _wait(api, ref: str, started: float, timeout: float, say) -> str:
    """Poll until the kernel stops, or we give up on it."""
    last = ""
    while True:
        waited = time.monotonic() - started
        if waited > timeout:
            raise RemoteError(
                f"kaggle job did not finish within {timeout / 60:.0f} minutes")
        try:
            status = api.kernels_status(ref)
        except Exception as exc:  # a transient API blip must not fail the job
            log.info("status check failed, retrying: %s", exc)
            time.sleep(POLL_SECONDS)
            continue

        state = str(status.get("status") if isinstance(status, dict)
                    else getattr(status, "status", "")).lower()
        if state != last:
            log.info("kaggle %s: %s (%.0fs)", ref, state, waited)
            last = state
        # Against the timeout, because Kaggle exposes no real progress.
        say("running", int(waited), int(timeout))

        if "complete" in state:
            return state
        if "error" in state or "cancel" in state:
            message = (status.get("failureMessage") if isinstance(status, dict)
                       else getattr(status, "failure_message", "")) or state
            raise RemoteError(f"kaggle job failed: {message}")
        time.sleep(POLL_SECONDS)


def _collect(api, ref: str, work: Path, state: str) -> dict:
    """Read the kernel's log and pull the result line out of it."""
    out = work / "out"
    out.mkdir(exist_ok=True)
    try:
        api.kernels_output(ref, str(out))
    except Exception as exc:
        raise RemoteError(f"could not read the job's output: {exc}") from exc

    text = "\n".join(p.read_text(errors="replace")
                     for p in out.rglob("*") if p.is_file())
    return parse_result(text)


def parse_result(text: str) -> dict:
    """Find the marked result line in a kernel log.

    The log wraps stdout in JSON records, so the marker arrives escaped. The
    last occurrence wins: a retried cell leaves more than one.
    """
    matches = re.findall(RESULT_MARKER + r"(?:\\n)?\s*(\{.*?\})\s*(?:\\n|\Z|\")",
                         text, re.DOTALL)
    if not matches:
        raise RemoteError("the job produced no result "
                          "(it may have run out of memory or time)")
    raw = matches[-1]
    # Undo the log wrapper's escaping, and only that: \n is a valid escape
    # inside JSON, so unescaping it here would break multi-line messages.
    raw = raw.replace('\\"', '"').replace("\\\\", "\\")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RemoteError(f"the job's result was not valid JSON: {exc}") from exc


# The kernel body, kept here so what runs remotely sits beside what calls it.
# It unpacks the bundle and hands off to the same functions the local path
# uses, so there is no second implementation to keep in step.
_KERNEL_SOURCE = r'''
import base64, io, json, os, sys, tarfile, traceback, time

_T0 = time.time()

# --- unpack the code this job was pushed with --------------------------------
_WORK = "/kaggle/working/vs"
os.makedirs(_WORK, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("%(bundle)s")), mode="r:gz") as _tar:
    _tar.extractall(_WORK)
sys.path.insert(0, _WORK)

PAYLOAD = json.loads(base64.b64decode("%(payload)s").decode())
KIND = "%(kind)s"
MARKER = "%(marker)s"

# --- dependencies Kaggle does not ship ----------------------------------------
# A few seconds against a job measured in minutes, so not worth vendoring.
_NEED = %(pips)s
if _NEED:
    import subprocess
    # Pin whatever torch Kaggle installed: a dependency pulling its own build
    # gets one compiled for different GPUs, which fails much later and a long
    # way from the pip line that caused it.
    _pins = "/tmp/vs-constraints.txt"
    try:
        import torch, torchvision
        with open(_pins, "w") as _f:
            _f.write(f"torch=={torch.__version__}\n")
            _f.write(f"torchvision=={torchvision.__version__}\n")
        _args = ["-c", _pins]
        print(f"pinning torch {torch.__version__}", flush=True)
    except Exception:
        _args = []
    print("installing: " + " ".join(_NEED), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "--disable-pip-version-check", *_args, *_NEED], check=False)
    try:
        import torch
        print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    except Exception as _e:
        print(f"torch unusable after install: {_e}", flush=True)

def _finish(obj):
    # Marked, because the log holds everything the notebook wrote.
    print(MARKER + " " + json.dumps(obj), flush=True)

# --- credentials --------------------------------------------------------------
# From the attached dataset, whose depth depends on how it was uploaded.
_CONFIG = {}
for _cfg in ("/kaggle/input/vs-config/config.json",
             "/kaggle/input/vs-config/vs-config/config.json"):
    if os.path.exists(_cfg):
        with open(_cfg) as _f:
            _CONFIG = json.load(_f)
        print(f"credentials <- {_cfg}")
        break
else:
    import glob as _glob
    for _cfg in _glob.glob("/kaggle/input/vs-config/**/config.json", recursive=True):
        with open(_cfg) as _f:
            _CONFIG = json.load(_f)
        print(f"credentials <- {_cfg}")
        break

def _secret(name, fallback=None):
    # Dataset first, then a Kaggle Secret, then whatever the job carried --
    # the last of which ends up in the stored notebook source, so it warns.
    if _CONFIG.get(name):
        return _CONFIG[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        if fallback:
            print(f"WARNING: {name} came from the job payload -- attach the "
                  f"vs-config dataset to keep it out of the notebook source")
        return fallback

os.environ["DATABASE_URL"] = _secret("DATABASE_URL", PAYLOAD.get("database_url", ""))

# YouTube refuses datacentre addresses more readily, and no player client gets
# past a bot challenge. A cookies.txt in vs-config is the way through.
for _ck in ("/kaggle/input/vs-config/cookies.txt",
            "/kaggle/input/vs-config/vs-config/cookies.txt"):
    if os.path.exists(_ck):
        os.environ["YTDLP_COOKIES"] = _ck
        print(f"youtube cookies <- {_ck}")
        break
else:
    if KIND == "index":
        print("no cookies.txt in vs-config; YouTube may refuse this address")
_gem = _secret("GEMINI_API_KEY", PAYLOAD.get("gemini_api_key"))
if _gem:
    os.environ["GEMINI_API_KEY"] = _gem

# --- point the pipeline at the attached weights -------------------------------
# Each dataset mounts under /kaggle/input/<slug>, turning a download into a
# disk read.
_INPUT = "/kaggle/input"
# Each variable may name several datasets; the first one attached wins. The
# captioner lists the pre-quantized build first because it is less than half
# the size and skips the quantize step, and the full checkpoint second so a
# kernel without the 4-bit dataset still works.
_LOCAL = {
    "SIGLIP_MODEL": ["vs-siglip2-so400m"],
    "BGE_MODEL": ["vs-bge-m3"],
    "CAPTION_MODEL": ["qwen3-vl-4b-4bit", "vs-qwen3-vl-4b"],
    "WHISPER_MODEL": ["vs-faster-whisper-turbo"],
}
for _var, _slugs in _LOCAL.items():
    _path = next((os.path.join(_INPUT, s) for s in _slugs
                  if os.path.isdir(os.path.join(_INPUT, s))), None)
    if _path is None:
        print(f"WARNING: none of {_slugs} attached; {_var} will be downloaded")
        continue

    # A model directory is the one holding config.json -- not "the one holding
    # any weights", which is what this used to look for. bge-m3 ships
    # pytorch_model.bin rather than safetensors, so the old test decided the
    # root was not a model and descended into its first subdirectory: the
    # sentence-transformers pooling config. The failure surfaced minutes later
    # as an unrelated-looking tokenizer error.
    if os.path.exists(os.path.join(_path, "config.json")):
        _model_dir = _path
    else:
        _subs = [os.path.join(_path, x) for x in sorted(os.listdir(_path))]
        _subs = [d for d in _subs
                 if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json"))]
        _model_dir = _subs[0] if _subs else _path
    os.environ[_var] = _model_dir
    print(f"{_var} <- {_model_dir}")

# Search must never try to use a GPU it was not given, and indexing must.
os.environ.setdefault("SEARCH_DEVICE", "cuda" if KIND == "index" else "cpu")

# Size the caption batch to the card actually allocated. A T4 is 15 GB against
# the 24 GB the default was chosen for, and the four models together leave
# little room: the difference showed up as an out-of-memory part-way through
# captioning, long after everything had loaded successfully.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
if KIND == "index":
    try:
        import torch as _t
        if _t.cuda.is_available():
            _gb = _t.cuda.get_device_properties(0).total_memory / 1e9
            # SigLIP and Whisper are released before captioning starts, so the
            # captioner has most of the card to itself; these thresholds assume
            # that. Without the release a T4 could only manage 4.
            _batch = 16 if _gb >= 22 else (12 if _gb >= 14 else 4)
            os.environ["CAPTION_BATCH"] = str(_batch)
            print(f"{_t.cuda.get_device_name(0)} {_gb:.0f} GB -> caption batch {_batch}")
    except Exception as _e:
        print(f"could not size the caption batch: {_e}")

try:
    if KIND == "search":
        from python.search import Searcher
        s = Searcher()
        result = s.handle(PAYLOAD["request"])
        result["kernel_seconds"] = round(time.time() - _T0, 1)
        _finish(result)
    elif KIND == "index":
        # The payload is small; the result is not. A 40-minute video produces
        # ~15k frame embeddings, which would be megabytes of base64 through a
        # notebook log. The kernel writes them to the database itself and
        # returns only what the caller needs to report.
        from python.index import index_url
        from python.lib.writeback import save    # the one writer, shipped in the bundle
        import psycopg

        # Say what the pipeline is doing, with elapsed time. Kaggle's notebook
        # page streams stdout while the kernel runs, so this is the only view
        # into a job that takes twenty minutes -- an earlier version discarded
        # these events and the notebook sat silent from model load to result,
        # which made a stall indistinguishable from slow progress.
        # The doubled sign below is deliberate: this whole block is a
        # percent-format template, so a single one is read as a substitution
        # and the push dies with KeyError before anything reaches Kaggle.
        # Comments are not exempt -- this note had to be reworded twice.
        import logging as _lg
        _lg.basicConfig(level=_lg.INFO, format="      %%(message)s", force=True)

        _last = {}

        def _say(stage, done=0, total=0, **_):
            # One line per stage, plus a tenth-of-the-way heartbeat. 2289
            # frames of "embed" is a flood, and silence is worse than a flood.
            if total:
                bucket = (done * 10) // total
                if done not in (0, total) and bucket == _last.get(stage):
                    return
                _last[stage] = bucket
            mark = f" {done}/{total}" if total else ""
            print(f"[{time.time() - _T0:6.0f}s] {stage}{mark}", flush=True)

        payload = index_url(PAYLOAD["url"], PAYLOAD["video_id"], on_progress=_say)
        print(f"[{time.time() - _T0:6.0f}s] writing to the database", flush=True)
        conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
        v = payload["video"]
        save(conn, PAYLOAD["video_id"], v["locator"], payload,
             source=PAYLOAD.get("source", "user"), title=v.get("title"))
        _finish({"ok": True, "video_id": PAYLOAD["video_id"],
                 "title": v.get("title"), "duration_s": v.get("duration_s"),
                 "locator": v["locator"], "stats": payload["stats"],
                 "kernel_seconds": round(time.time() - _T0, 1)})
    else:
        _finish({"ok": False, "error": f"unknown job kind {KIND}"})
except Exception as _exc:
    traceback.print_exc()
    _finish({"ok": False, "error": f"{type(_exc).__name__}: {_exc}"})
'''
