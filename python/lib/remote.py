"""Run a job on Kaggle's free GPU and bring the answer back.

The deployment target has no RAM for the models -- not even the two the search
path needs -- so the work happens on a Kaggle kernel instead. Go is unchanged:
it still runs search.py and index.py and still speaks the same JSON protocol.
Only what happens behind that protocol moves.

Two things are worth knowing before reading further.

**Every job is a fresh kernel.** Kaggle offers no way to hold a process open and
call into it, so a job is: push a notebook, poll until it finishes, read its
output. A no-op kernel measured 40 seconds from push to output on this account,
and that is the floor -- before any model loads. Search here costs minutes, not
the 285ms it costs locally. That is inherent to the approach, not a bug in it.

**The kernel carries its own code.** The notebook embeds a base64 tarball of
python/, so the kernel always runs exactly the code in this repo. The obvious
alternative -- keeping a "code" dataset on Kaggle and attaching it -- needs a
separate upload every time anything changes here, and silently runs stale code
when that upload is forgotten.

Weights are the exception: they come from attached datasets, because they are
gigabytes and they do not change. See models/README.md.
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

# The kernel prints its answer on one line with this marker. Kaggle's output is
# the whole notebook log -- every warning, every progress bar -- so the result
# has to be findable rather than "the last line".
RESULT_MARKER = "__VS_RESULT__"

KAGGLE_USER = os.environ.get("KAGGLE_USERNAME", "biganradu335ca")

# One slug per job kind, reused. Kaggle keeps versions, so pushing again is an
# update rather than a new notebook -- which matters because an account
# accumulating a kernel per search would be both unusable and rude.
SLUGS = {"search": "vs-search-worker", "index": "vs-index-worker"}

# Attached weights, per job kind. Search does not attach the captioner: it
# would add ~8 GB of mount time to a job that never opens it.
# vs-config carries the database URL. It has to be a dataset rather than a
# Kaggle Secret: secrets attached in the notebook editor are NOT preserved when
# a notebook is pushed through the API, and kernel-metadata.json has no field
# for them -- so with a fresh push per job, an attached secret is wiped every
# time. Dataset attachments do survive, because they are in that metadata.
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

# How long to wait before giving up, per kind. Indexing is a minute of GPU per
# minute of video; search should never be slow enough to reach its limit, so
# hitting it means something is wrong rather than merely slow.
TIMEOUTS = {"search": 15 * 60, "index": 3 * 60 * 60}

POLL_SECONDS = float(os.environ.get("KAGGLE_POLL_SECONDS", 10))

# What each kind needs that Kaggle's image lacks. Deliberately short: every
# entry is seconds added to every job, and torch/transformers are already there.
#
# sentencepiece is not optional despite tokenizer.json being present: on
# Kaggle's transformers, bge-m3's XLM-R tokenizer falls back to converting the
# slow tokenizer and fails without it. It worked when the model came straight
# from Hugging Face and broke when it came from an attached dataset, which made
# it look like a bad upload -- the files were byte-identical.
PIPS = {
    "search": ["psycopg[binary]", "sentencepiece", "protobuf"],
    # bitsandbytes is what loads the captioner in 4-bit. Kaggle's image does
    # not carry it, and transformers only says so once it reaches the load.
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

    Imported lazily: the package authenticates on import and raises when no
    credentials are present, which must not happen merely because something
    imported this module on a machine that never talks to Kaggle.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _bundle() -> str:
    """python/ as a base64 tarball, for the notebook to unpack.

    Only .py files, and not __pycache__: the bundle is embedded in a JSON
    notebook that is uploaded on every single job, so its size is paid over and
    over. The whole tree compresses to tens of kilobytes.
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

    Kept to a single cell on purpose. A multi-cell notebook that fails in cell 2
    still "completes", and the failure has to be inferred from the absence of
    output; one cell either raises or prints a result.
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
        # Ask for a T4, never a P100. Kaggle's image ships torch built for
        # CUDA 12.8, whose kernels start at sm_70 -- the P100 is sm_60, so a
        # job that lands on one dies inside the first forward pass with
        # "no kernel image is available for execution on the device". Without
        # this, which GPU you get is luck.
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
        # Progress is reported against the timeout because Kaggle exposes no
        # notion of how far along a running kernel is.
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

    The log is JSON-wrapped stream records, so the marker arrives escaped and
    cannot simply be split on. Searching for the last occurrence matters: a
    retried cell would leave two, and the later one is the live answer.
    """
    matches = re.findall(RESULT_MARKER + r"(?:\\n)?\s*(\{.*?\})\s*(?:\\n|\Z|\")",
                         text, re.DOTALL)
    if not matches:
        raise RemoteError("the job produced no result "
                          "(it may have run out of memory or time)")
    raw = matches[-1]
    # Undo the JSON-string escaping the log wrapper applied -- and only that.
    # An earlier version also turned \n into a real newline, which corrupted any
    # result carrying a multi-line message (an ffmpeg error, say): \n is a valid
    # escape *inside* JSON, and rewriting it puts a raw control character in a
    # string. The backslash rule below already yields the right thing.
    raw = raw.replace('\\"', '"').replace("\\\\", "\\")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RemoteError(f"the job's result was not valid JSON: {exc}") from exc


# The kernel body. Written as a template rather than a file so that what runs
# remotely is versioned here beside what calls it.
#
# It unpacks the bundle, puts it on sys.path, and hands off to the same
# functions the local path uses -- there is no second implementation of search
# or indexing to keep in step.
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
# Kaggle's image has torch and transformers, which are the big ones, but it
# ships psycopg2 rather than psycopg 3 -- and this codebase uses 3. Installing
# is a few seconds against a job measured in minutes, so it is not worth
# vendoring. --quiet because pip's output is longer than the job's.
_NEED = %(pips)s
if _NEED:
    import subprocess
    # Pin whatever torch Kaggle already installed. faster-whisper and friends
    # pull their own torch otherwise, and the replacement is built for
    # different compute capabilities than the GPU in the box -- which surfaces
    # much later as "CUDA error: no kernel image is available for execution on
    # the device", a long way from the pip line that caused it.
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
    # One line, marked. Kaggle's log is everything the notebook wrote, so the
    # answer has to be findable rather than positional.
    print(MARKER + " " + json.dumps(obj), flush=True)

# --- credentials --------------------------------------------------------------
# Kaggle Secrets first: a DSN embedded in the notebook would be stored on
# Kaggle in the source of every job ever pushed. The payload fallback exists so
# this works before the secret is configured, and warns when it is used.
# Where the file lands depends on how the dataset was made: uploading a folder
# keeps its name, so config.json can sit one level down. Both are normal, so
# look rather than assume a path.
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
    # The attached dataset first, then a Kaggle Secret for anyone running this
    # notebook by hand, then whatever the job carried. The last of those puts
    # the value in the stored source of every pushed version, so it warns.
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

# YouTube refuses datacentre addresses far more readily than home ones, and
# once it asks to "confirm you're not a bot" no player client gets past it.
# A cookies.txt placed in the vs-config dataset is the way through.
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
# Each dataset mounts under /kaggle/input/<slug>. Setting the model paths to
# those directories is what turns a multi-gigabyte download into a disk read.
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
