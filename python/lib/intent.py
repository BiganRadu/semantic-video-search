"""Plan a query before searching: what it is asking for, and how to phrase it
for each index.

Two things from one model call.

**Routing.** One global weight vector cannot serve every query. "a woman
talking to the camera" is answered by the pixels; "a woman explaining why she
quit" is answered by the transcript. The class picks the fusion weights.

**Rewriting.** Each index is searched by embedding text, so the query should
look like what it is compared against -- a different text per index:

    "the part where the guy in the blue shirt jumps off the wall"
      visual   -> "a man in a blue shirt jumping off a wall"
      caption  -> "A man wearing a blue shirt leaps from the top of a wall."
      keywords -> ["blue shirt", "wall", "jump"]

Speech is the interesting one: a transcript holds what someone *said*, not a
description of them saying it, so the useful thing to embed is a hypothetical
utterance (HyDE) rather than the query itself.

Bounded by a timeout, and every failure degrades to the raw query with the
default weights -- a dead endpoint slows search but never breaks it.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field

from python.lib.config import (GEMINI_API_KEY, INTENT_BASE_URL, INTENT_MODEL,
                               INTENT_TIMEOUT)

# What each class does to the fusion weights. Starting points rather than
# measured results, with one exception.
#
# caption is 0 on a speech query and has to be: caption_vec is written by the
# vision model from frames alone, so it cannot contain what anyone said and can
# only contribute confident noise. keyword survives for the opposite reason --
# clips.fts covers speech as well, making it the one literal path into the
# transcript.
PROFILES = {
    "visual": {"visual": 1.0, "caption": 1.0, "speech": 0.0, "keyword": 0.0},
    "speech": {"visual": 0.0, "caption": 0.0, "speech": 1.0, "keyword": 0.5},
    "mixed":  {"visual": 1.0, "caption": 1.0, "speech": 0.6, "keyword": 0.25},
}
CLASSES = tuple(PROFILES)

SYSTEM = """You prepare queries for a video search engine. A clip is indexed three ways,
and each index is searched by embedding text and comparing vectors, so the text
you write should look like the thing being searched.

  visual   - CLIP-style image embeddings. Matches short, concrete, descriptive
             phrases. Write what a camera would see. No questions, no "find me".
  caption  - sentences a vision model wrote about each clip, e.g. "A woman in a
             red coat walks through a market and picks up an apple."
  speech   - a transcript of the words spoken. Write what a person would
             actually SAY about this, in their own voice - not a description of
             them saying it.
  keywords - rare or literal terms only: names, places, numbers, brands. These
             go to a keyword index that embeddings are bad at. [] if none.

Also classify where the ANSWER lives:
  speech - the query is about what someone SAYS (topics, claims, explanations)
  visual - the query describes what would be SEEN
  mixed  - needs both, or ambiguous
"a woman talking to the camera" is VISUAL - you recognise it by looking.
"a woman talking about her divorce" is SPEECH - the topic is only in the words.

Reply with JSON only:
{"class":"visual|speech|mixed","confidence":0.0-1.0,
 "visual":"...","caption":"...","speech":"...","keywords":["..."]}"""


@dataclass(slots=True)
class Plan:
    cls: str
    confidence: float
    # Per-signal query text. A signal missing here falls back to the raw query.
    queries: dict = field(default_factory=dict)
    took_ms: float = 0.0

    def weights(self) -> dict:
        return dict(PROFILES[self.cls])

    def text_for(self, signal: str, fallback: str) -> str:
        return (self.queries.get(signal) or "").strip() or fallback

    def as_json(self) -> dict:
        return {"class": self.cls, "confidence": self.confidence,
                "queries": self.queries, "took_ms": round(self.took_ms, 1)}


def enabled() -> bool:
    return bool(GEMINI_API_KEY)


def log(message: str) -> None:
    """stderr, never stdout -- stdout carries the JSON protocol Go reads.

    Failures are swallowed so a search cannot break, but swallowed and silent
    are different things: without this a dead endpoint looks like a slow one.
    """
    print(f"[intent] {message}", file=sys.stderr, flush=True)


class Router:
    """Query -> weights and per-index phrasing. One call, synchronous."""

    def __init__(self, conn=None) -> None:
        self._client = None

    def plan(self, query: str) -> Plan | None:
        """Plan one query, or None if routing is off or the model did not answer."""
        if not enabled() or not (query or "").strip():
            return None

        client = self._openai()
        if client is None:
            return None

        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=INTENT_MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": query}],
                temperature=0,
                # Room for four rewrites; thinking off, because this is a
                # formatting and classification job, not a reasoning one.
                max_tokens=320,
                # No reasoning_effort: this endpoint rejects the field with a
                # 400, and flash-lite answers in under a second without it.
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            took = (time.monotonic() - started) * 1000
            log(f"plan failed after {took:.0f}ms: {type(exc).__name__}: {str(exc)[:160]}")
            return None

        took = (time.monotonic() - started) * 1000
        plan = parse_plan(response.choices[0].message.content)
        if plan is None:
            log(f"{query[:50]!r} -> unparseable after {took:.0f}ms")
            return None
        plan.took_ms = took
        log(f"{query[:50]!r} -> {plan.cls} ({took:.0f}ms) "
            f"rewrote {sorted(plan.queries)}")
        return plan

    def _openai(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                return None
            self._client = OpenAI(base_url=INTENT_BASE_URL, api_key=GEMINI_API_KEY,
                                  timeout=INTENT_TIMEOUT, max_retries=0)
        return self._client


def parse_plan(text: str | None) -> Plan | None:
    """Read the model's answer, defensively.

    A model is not a parser: it wraps JSON in prose, in fences, or answers with
    a bare word. An unrecognised class is rejected outright -- routing on a
    class we have no profile for would silently retrieve nothing.
    """
    if not text:
        return None
    body = re.search(r"\{.*\}", text.strip(), re.S)
    if not body:
        return None
    try:
        data = json.loads(body.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    cls = str(data.get("class", "")).strip().lower()
    if cls not in CLASSES:
        return None

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 1.0))))
    except (TypeError, ValueError):
        confidence = 1.0

    queries: dict[str, str] = {}
    for signal in ("visual", "caption", "speech"):
        value = data.get(signal)
        if isinstance(value, str) and value.strip():
            # A rewrite longer than this is the model narrating rather than
            # phrasing; SigLIP's text tower only reads 64 tokens anyway.
            queries[signal] = value.strip()[:400]

    words = data.get("keywords")
    if isinstance(words, str):
        words = [words]
    if isinstance(words, list):
        terms = [str(w).strip() for w in words if str(w).strip()]
        if terms:
            # keyword_candidates extracts lexemes and ORs them, so joining is
            # all that is needed -- no tsquery syntax is ever constructed here.
            queries["keyword"] = " ".join(terms)[:400]

    return Plan(cls=cls, confidence=confidence, queries=queries)
