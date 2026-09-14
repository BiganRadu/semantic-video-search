#!/usr/bin/env python3
"""Search the index.

Go starts this script once at boot and talks to it over stdin/stdout: one JSON
request per line in, one JSON response per line out. It stays warm because a
cold start costs about 9.5 seconds -- 1.4s to import torch, 3.5s transformers,
4.1s for the first forward pass -- which would otherwise be paid on every
single query.

    >>> {"q": "a person walking a dog", "k": 10}
    <<< {"ok": true, "results": [...], "corpus": {...}, "took_ms": 14.2}

Results are *moments*, not clips: adjacent clips of one video that all matched
are assembled back into the span they cover, so one event appears once.

Also runs one-shot for debugging:  python/search.py --query "a red car"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.lib.config import CLIP_LEN, DATABASE_URL, VISUAL_DIM, device_for
from python.lib.models import BGEM3, SigLIP, vector_literal

# How deep each signal goes before fusion. Fusion and reranking need more
# candidates than the caller asked for.
CANDIDATE_DEPTH = 200

# Signals that exist in principle. Only IMPLEMENTED ones return data; asking
# for the others fails loudly rather than silently returning a subset, so an
# ablation can never quietly measure the wrong thing.
KNOWN = ("visual", "caption", "speech", "keyword")
IMPLEMENTED = ("visual", "caption", "speech", "keyword")

# How many results a page shows, by scope, when the caller does not say.
#
# A within-video search answers "where in this video", and three timestamps is
# an answer -- twenty is the video handed back. Corpus search shows more because
# breadth is the point there. The relevance floor can still return fewer, and
# often should: "nothing here matches" is a real answer.
DEFAULT_K = {"corpus": 8, "video": 3}

# Damping constant from the original reciprocal-rank-fusion paper. Not tuned.
RRF_K = 60.0

# Per-signal fusion weights.
#
# These are NOT optional. Unweighted RRF gives every signal an equal vote, so a
# clip ranked first by a near-useless signal outranks nothing -- it actively
# displaces one ranked first by a good signal. Measured on dev100, unweighted
# fusion of all three signals scored R@1 0.270 against 0.658 for visual alone.
#
# A weight of 0 disables a signal's contribution without removing it from the
# response, which is useful for seeing what a signal *would* have retrieved.
#
# These values are MEASURED, not guessed: a 64-point sweep on the QVHighlights
# train split (140 queries, held out from everything reported) put the optimum
# at speech=0 and keyword=0. Every non-zero weight scored worse on nDCG@10.
# See eval/reports/weights_train100.json.
#
# That is a statement about QVHighlights, not about speech in general. Its
# queries describe what is visible ("a girl is showing her hotel room") while
# the audio is unrelated vlog chatter, so transcript matching only adds noise.
# On footage where people say what they are doing -- meetings, lectures,
# security audio -- the tuning would land somewhere else. Retune per corpus.
# caption starts at 1.0 pending its own sweep; the others are the measured
# optimum from the pre-caption tuning run.
DEFAULT_WEIGHTS = {"visual": 1.0, "caption": 1.0, "speech": 0.0, "keyword": 0.0}

# -- moment assembly ---------------------------------------------------------
#
# Retrieval scores 10s clips because that is the granularity the models see, but
# a clip is not what anyone searched for. One event -- someone carrying a box out
# of a door -- spans several clips and every one of them matches, so an
# unassembled page shows the same event five times and pushes the rest of the
# corpus off it. Assembly merges adjacent retrieved clips of one video back into
# the span they actually cover, and ranks spans instead of clips.

# Largest hole bridged when joining two retrieved clips. Measured inert at the
# cap below -- bridging a 10s hole needs a 30s span and the cap is 20s -- so it
# is off. It exists for corpora whose events run longer than this one's.
MOMENT_GAP = 0.0

# A moment stops growing here, and this one is MEASURED. At 60s, moments came
# back averaging 53.6s -- a third of a 150s video -- and scored *better* on "the
# returned range touches a relevant window" while getting visibly worse, because
# a long enough span touches something by accident. Against the stricter "half
# the returned range is inside the window" it halved: R@1 0.279 -> 0.135.
#
# A 140-query sweep on the train split put every setting that holds retrieval
# quality at 20s, and the ground truth says why: the median QVHighlights window
# is 14s and 65% are 20s or shorter. An answer should be about as long as the
# thing it is an answer to.
MOMENT_MAX_LEN = 20.0

# What corroboration from the rest of a moment is worth. Clip scores are summed
# in descending order under a geometric discount, so a moment's total is bounded
# by peak / (1 - decay): supporting clips can lift a moment's rank, but length
# alone can never win. At 0 a moment scores exactly its best clip and assembly
# becomes pure de-duplication; 0.5 measured better than 0 on the train split
# (R@1 0.586 against 0.543) at no cost to the strict metric.
MOMENT_DECAY = 0.5

# How strong a neighbouring clip must be, relative to the moment's peak, to be
# absorbed into it. Without this a moment grows into every clip the candidate
# pool happens to contain for that video, which at CANDIDATE_DEPTH=200 is most
# of it -- and a span that long scores well on "touches a relevant window" for
# the wrong reason. This is the parameter that keeps a moment tight.
# Measured inert at a 20s cap -- a two-clip moment rarely has a second clip weak
# enough to refuse -- and kept because it is what would hold a longer moment
# together if the cap were raised. Values above 0.5 measured worse.
MOMENT_FLOOR = 0.0

# How many moments one video may hold on the page before the rest are pushed
# below everything else. Merging alone does not fix a flooded page: a video with
# three genuinely separate matching events legitimately produces three moments,
# and on a 150s clip they are often the same event anyway. Demoted moments are
# not dropped -- a page short of k fills back up from them in score order -- so
# this changes the order of the answers, never which answers exist. 0 disables.
MOMENT_PER_VIDEO = 2

# Below these raw similarities, a moment is not a match and is not returned.
#
# This exists because RRF scores rank, not similarity, and rank is meaningless
# when there is nothing to lose to. Searching inside ONE video, every clip is in
# the candidate list, so every clip gets a reciprocal rank and the top always
# scores about 1/(RRF_K+1) -- whatever the query. Measured: "a man is talking"
# and "zzzz nonsense qqq" against the same video returned 9 moments each, with
# the same scores. The timeline lit up identically for both.
#
# The raw cosines do know the difference, and they separate cleanly. Over 40
# QVHighlights queries against their own video, versus 40 random-letter queries
# against those same videos:
#
#     caption   real 0.480-0.781 (median 0.649)   nonsense 0.280-0.403
#     visual    real 0.103-0.229 (p10-max)        nonsense 0.013-0.092
#
# caption does not overlap at all. The floors sit just above what nonsense
# reached, so a real match is kept and a non-match is dropped.
#
# A moment survives if ANY signal clears its own floor -- one strong signal is
# enough, and the scales are not comparable to each other.
#
# keyword has no floor: ts_rank_cd only scores rows that already matched the
# tsquery, so the signal is its own filter.
MIN_RELEVANCE = {"visual": 0.09, "caption": 0.42, "speech": 0.42, "keyword": 0.0}

# The other end of the same measurement: what a strong match actually scores.
# These are the p90 of real queries against their own video, from the same run.
# Together with MIN_RELEVANCE they turn a raw cosine into a 0..1 confidence
# that means the same thing across signals whose scales do not compare --
# 0 is "no better than nonsense", 1 is "as good as this signal gets".
#
# This is what the interface should show, not the fusion score. RRF ranks; it
# does not measure. Inside one video the top moment always scores about
# 1/(RRF_K+1) whether or not anything matched, so a meter driven by it is a
# meter that always reads full.
MAX_RELEVANCE = {"visual": 0.17, "caption": 0.72, "speech": 0.72, "keyword": 1.0}

# Clip boundaries come back from Postgres as floats and the last clip of a video
# is a folded short tail, so adjacency is compared with a tolerance rather than
# by equality.
EPS = 1e-6


def as_bool(value, default: bool) -> bool:
    """Coerce a request field to a bool.

    Go forwards query parameters as strings, so `assemble=0` arrives as "0",
    which is truthy in Python. Anything unrecognised keeps the default rather
    than guessing.
    """
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def as_float(value, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class Scope:
    """Which slice of the index a request may see.

    A visitor's videos and the shared example corpus are separate worlds: a
    search never crosses between them, so one person's uploads can never appear
    in someone else's results.
    """

    __slots__ = ("collection", "owner", "video_id")

    def __init__(self, collection: str = "examples", owner: str | None = None,
                 video_id: str | None = None):
        self.collection = collection
        # The namespaced owner key exactly as it appears in videos.owner --
        # "anon:<session>" or "user:<id>", never a bare id. Go composes it; this
        # side only ever compares it, so there is one place that knows the
        # format and it is not this one.
        self.owner = owner
        self.video_id = video_id


class Searcher:
    def __init__(self) -> None:
        import psycopg

        self.device = device_for("search")
        self._siglip = None
        self._bge = None
        self.conn = psycopg.connect(DATABASE_URL, autocommit=True)

        # Query routing, if a key is configured. Nothing downstream depends on
        # it: with no router the weights are simply the measured defaults.
        from python.lib.intent import Router
        self.router = Router(self.conn)

    # Encoders load on first use, not at startup. A deployment whose weights
    # disable the caption and speech signals never touches bge-m3, and that is
    # 1.5 GB of resident memory on a box that may only have two. Measured:
    # 2.7 GB visual-only against 4.2 GB with both loaded.

    @property
    def siglip(self) -> SigLIP:
        if self._siglip is None:
            self._siglip = SigLIP(self.device, vision=False)
        return self._siglip

    @property
    def bge(self) -> BGEM3:
        if self._bge is None:
            self._bge = BGEM3(self.device)
        return self._bge

    # -- scoping -----------------------------------------------------------

    @staticmethod
    def corpus_filter(scope: Scope) -> tuple[str, list]:
        """SQL restricting a search to one corpus, and optionally one video.

        Only literal SQL is interpolated; every value travels as a parameter.
        The example corpus is the rows with no owner, so a visitor's uploads are
        never mixed into it and vice versa.
        """
        clauses, params = [], []

        if scope.collection == "mine":
            # An empty owner matches nothing rather than everything: a request
            # with no session must not fall through to somebody else's corpus.
            clauses.append("v.owner = %s")
            params.append(scope.owner or "\x00 no owner")
        else:
            clauses.append("v.owner IS NULL")

        if scope.video_id:
            clauses.append("c.video_id = %s")
            params.append(scope.video_id)

        # Expired session videos stay in the table until swept, but stop being
        # searchable the moment they lapse.
        clauses.append("(v.expires_at IS NULL OR v.expires_at > now())")

        return " AND " + " AND ".join(clauses), params

    # -- retrieval ---------------------------------------------------------

    def visual_candidates(self, query: str, scope: "Scope") -> list[dict]:
        """Rank clips by cosine similarity in the SigLIP space.

        The scope filter is an ordinary WHERE clause. With no ANN index this is
        an exact scan, so a scoped search is both faster (the btree on
        (video_id, start_s) cuts it to one video) and exact.
        """
        vec = vector_literal(self.siglip.embed_texts([query])[0])
        where, scope_params = self.corpus_filter(scope)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT c.id, c.video_id, c.start_s, c.end_s, v.locator,
                       1 - (cv.visual <=> %s::vector) AS similarity
                FROM clip_vectors cv
                JOIN clips  c ON c.id = cv.clip_id
                JOIN videos v ON v.id = c.video_id
                WHERE cv.visual IS NOT NULL {where}
                ORDER BY cv.visual <=> %s::vector
                LIMIT %s
                """,
                [vec, *scope_params, vec, CANDIDATE_DEPTH],
            )
            return self._rows(cur)

    def speech_candidates(self, query: str, scope: "Scope") -> list[dict]:
        """Rank clips by transcript similarity, using bge-m3 rather than SigLIP.

        Two embedders on purpose: SigLIP is strong image-to-text and weak
        text-to-text, so speech (and later captions) are matched with a model
        actually trained for that.
        """
        vec = vector_literal(self.bge.embed_texts([query])[0])
        where, scope_params = self.corpus_filter(scope)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT c.id, c.video_id, c.start_s, c.end_s, v.locator,
                       1 - (cv.speech_vec <=> %s::vector) AS similarity
                FROM clip_vectors cv
                JOIN clips  c ON c.id = cv.clip_id
                JOIN videos v ON v.id = c.video_id
                WHERE cv.speech_vec IS NOT NULL {where}
                ORDER BY cv.speech_vec <=> %s::vector
                LIMIT %s
                """,
                [vec, *scope_params, vec, CANDIDATE_DEPTH],
            )
            return self._rows(cur)

    def caption_candidates(self, query: str, scope: "Scope") -> list[dict]:
        """Rank clips by generated-caption similarity, via bge-m3.

        The caption vector covers the caption plus its flattened tags, so a
        query naming an object matches even when the sentence does not.
        """
        vec = vector_literal(self.bge.embed_texts([query])[0])
        where, scope_params = self.corpus_filter(scope)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT c.id, c.video_id, c.start_s, c.end_s, v.locator,
                       1 - (cv.caption_vec <=> %s::vector) AS similarity
                FROM clip_vectors cv
                JOIN clips  c ON c.id = cv.clip_id
                JOIN videos v ON v.id = c.video_id
                WHERE cv.caption_vec IS NOT NULL {where}
                ORDER BY cv.caption_vec <=> %s::vector
                LIMIT %s
                """,
                [vec, *scope_params, vec, CANDIDATE_DEPTH],
            )
            return self._rows(cur)

    def keyword_candidates(self, query: str, scope: "Scope") -> list[dict]:
        """Literal matching over the generated tsvector.

        Embeddings blur, which is right for "someone leaves" ~ "exits through
        the gate" and wrong for proper nouns, numbers and rare terms. This is
        the branch that catches those.

        The terms are OR-ed, not AND-ed. websearch_to_tsquery and
        plainto_tsquery both AND, which turns this into a filter: a natural
        language query like "she talks about her hotel room" would demand one
        clip containing every term and return nothing. As a signal feeding
        fusion it should rank by how well a clip matches, so the lexemes are
        extracted through to_tsvector -- which stems them and drops stop words
        -- and joined with OR.

        Going through to_tsvector also means no user input is ever concatenated
        into a tsquery.
        """
        where, scope_params = self.corpus_filter(scope)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                WITH q AS (
                    SELECT to_tsquery('english',
                        nullif(array_to_string(
                            tsvector_to_array(to_tsvector('english', %s)), ' | '), '')
                    ) AS tsq
                )
                SELECT c.id, c.video_id, c.start_s, c.end_s, v.locator,
                       ts_rank_cd(c.fts, q.tsq) AS rank
                FROM clips  c
                JOIN videos v ON v.id = c.video_id
                CROSS JOIN q
                WHERE q.tsq IS NOT NULL AND c.fts @@ q.tsq {where}
                ORDER BY rank DESC
                LIMIT %s
                """,
                [query, *scope_params, CANDIDATE_DEPTH],
            )
            return self._rows(cur)

    @staticmethod
    def _rows(cur) -> list[dict]:
        return [
            {"clip_id": r[0], "video_id": r[1], "start_s": r[2],
             "end_s": r[3], "locator": r[4], "raw": float(r[5])}
            for r in cur.fetchall()
        ]

    def corpus_size(self, scope: "Scope") -> dict:
        """What was actually searched. Recall is meaningless without it, and a
        visitor should see the size of their own corpus, not the whole table."""
        where, params = self.corpus_filter(scope)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT count(DISTINCT c.video_id), count(*)
                    FROM clips c JOIN videos v ON v.id = c.video_id
                    WHERE true {where}""",
                params,
            )
            videos, clips = cur.fetchone()
            return {"videos": videos, "clips": clips}

    def coverage(self, scope: "Scope" = None) -> dict:
        """How much of the corpus each signal actually covers.

        A signal that scores badly because it has no data is a different
        finding from one that scores badly with data, and the ablation table
        cannot tell them apart on its own.
        """
        where, params = self.corpus_filter(scope or Scope())
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT count(*) FILTER (WHERE cv.visual IS NOT NULL),
                           count(*) FILTER (WHERE cv.caption_vec IS NOT NULL),
                           count(*) FILTER (WHERE cv.speech_vec IS NOT NULL),
                           count(*) FILTER (WHERE c.speech IS NOT NULL OR c.caption IS NOT NULL),
                           count(*)
                    FROM clips c
                    JOIN videos v ON v.id = c.video_id
                    LEFT JOIN clip_vectors cv ON cv.clip_id = c.id
                    WHERE true {where}""",
                params,
            )
            visual, caption, speech, searchable_text, total = cur.fetchone()
            return {"visual": visual, "caption": caption, "speech": speech,
                    "keyword": searchable_text, "clips": total}

    # -- ranking -----------------------------------------------------------

    @staticmethod
    def fuse(lists: dict[str, list[dict]], k: int | None = None,
             weights: dict[str, float] | None = None) -> list[dict]:
        """Weighted Reciprocal Rank Fusion.

        RRF uses only rank, never score. SigLIP cosines, bge-m3 cosines and
        ts_rank_cd sit on scales that are not comparable and that shift
        whenever a model is swapped; a weighted sum of them would need
        recalibrating every time. Rank fusion needs no calibration.

        The weights are what stop a weak signal from outvoting a strong one --
        see DEFAULT_WEIGHTS. With one signal the order is unchanged whatever the
        weight, since 1/(k+rank) is monotone in rank. Raw per-signal scores are
        carried through untouched.

        k=None returns the whole fused pool, which is what assembly consumes:
        moments have to be built from every retrieved clip, because truncating
        first would cut a moment off at the page boundary.
        """
        weights = weights or DEFAULT_WEIGHTS
        merged: dict[int, dict] = {}
        for signal, cands in lists.items():
            weight = weights.get(signal, 1.0)
            for rank, c in enumerate(cands, start=1):
                row = merged.setdefault(c["clip_id"], {
                    "clip_id": c["clip_id"], "video_id": c["video_id"],
                    "start_s": c["start_s"], "end_s": c["end_s"],
                    "locator": c["locator"], "signals": {}, "score": 0.0,
                })
                row["signals"][signal] = c["raw"]
                row["score"] += weight / (RRF_K + rank)

        # Deterministic ties: same query, same order, every run.
        out = sorted(merged.values(),
                     key=lambda r: (-r["score"], r["video_id"], r["start_s"]))
        return out if k is None else out[:k]


    # -- moment assembly ---------------------------------------------------

    @staticmethod
    def assemble(fused: list[dict], k: int | None = None,
                 gap: float = MOMENT_GAP,
                 max_len: float = MOMENT_MAX_LEN,
                 decay: float = MOMENT_DECAY,
                 floor: float = MOMENT_FLOOR,
                 per_video: int = MOMENT_PER_VIDEO,
                 min_relevance: dict | None = None) -> list[dict]:
        """Merge adjacent retrieved clips of one video into ranked moments.

        Greedy, seeded from the best clip outward: take the highest-scoring clip
        no moment has claimed, then grow it towards whichever neighbour scores
        better, stopping when the neighbours run out, a hole wider than `gap`
        appears, or the span would exceed `max_len`.

        Growing outward from the peak rather than merging runs left to right is
        what makes the cap behave: whatever `max_len` cuts off is the weak end,
        so a moment stays centred on its own evidence. And because seeds are
        consumed in descending score order, a moment's seed is always its own
        best clip and no clip can belong to two moments -- the property that
        actually removes the duplicates.

        A neighbour also has to be worth absorbing: `floor` is the fraction of
        the peak's score it must reach. Without it a moment swallows every clip
        of its video that appears anywhere in the candidate pool, and a span
        that long scores well on "touches a relevant window" for the wrong
        reason -- which is exactly the kind of improvement this project is
        supposed to catch rather than report.

        Assembly only ever reorders and groups what fusion retrieved. It cannot
        introduce a clip no signal matched, so it cannot manufacture a hit.
        """
        by_video: dict[str, list[dict]] = {}
        for clip in fused:
            by_video.setdefault(clip["video_id"], []).append(clip)
        for clips in by_video.values():
            clips.sort(key=lambda c: c["start_s"])

        # Where each clip sits on its video's retrieved timeline, so finding a
        # neighbour is a lookup rather than a scan.
        position = {c["clip_id"]: (c["video_id"], i)
                    for clips in by_video.values()
                    for i, c in enumerate(clips)}

        claimed: set[int] = set()
        moments: list[dict] = []

        for seed in fused:                       # already in descending score
            if seed["clip_id"] in claimed:
                continue
            video_id, i = position[seed["clip_id"]]
            clips = by_video[video_id]
            lo = hi = i
            claimed.add(seed["clip_id"])
            cutoff = floor * seed["score"]

            while True:
                left = clips[lo - 1] if lo > 0 else None
                right = clips[hi + 1] if hi + 1 < len(clips) else None

                if left is not None and (
                        left["clip_id"] in claimed
                        or left["score"] < cutoff
                        or clips[lo]["start_s"] - left["end_s"] > gap + EPS
                        or clips[hi]["end_s"] - left["start_s"] > max_len + EPS):
                    left = None
                if right is not None and (
                        right["clip_id"] in claimed
                        or right["score"] < cutoff
                        or right["start_s"] - clips[hi]["end_s"] > gap + EPS
                        or right["end_s"] - clips[lo]["start_s"] > max_len + EPS):
                    right = None

                if left is None and right is None:
                    break
                if right is None or (left is not None
                                     and left["score"] >= right["score"]):
                    lo -= 1
                    claimed.add(clips[lo]["clip_id"])
                else:
                    hi += 1
                    claimed.add(clips[hi]["clip_id"])

            moments.append(Searcher._moment(clips[lo:hi + 1], seed, decay))

        # A clip the floor refused is not lost: it is left unclaimed and becomes
        # the seed of its own moment on a later pass, which is why every
        # retrieved clip still ends up in exactly one moment.

        moments.sort(key=lambda m: (-m["score"], m["video_id"], m["start_s"]))
        moments = [m for m in moments if Searcher.relevant(m, min_relevance)]
        return Searcher.spread(moments, k, per_video)

    @staticmethod
    def confidence(signals: dict) -> float:
        """How much of a match this is, on a 0..1 scale, from raw similarity.

        The best any one signal managed, rescaled between "no better than a
        nonsense query" and "as good as this signal gets". Signals are compared
        after rescaling, never before: a 0.6 caption cosine and a 0.6 SigLIP
        cosine are not the same claim.
        """
        best = 0.0
        for name, raw in signals.items():
            lo = MIN_RELEVANCE.get(name)
            hi = MAX_RELEVANCE.get(name)
            if lo is None or hi is None or hi <= lo:
                continue
            best = max(best, min(1.0, max(0.0, (raw - lo) / (hi - lo))))
        return round(best, 4)

    @staticmethod
    def relevant(moment: dict, floors: dict | None) -> bool:
        """Is this a match at all, or merely the best of a bad lot?

        Rank cannot answer that -- see MIN_RELEVANCE. Raw similarity can, so a
        moment has to clear the floor on at least one signal. Returning nothing
        is a valid answer to a query the corpus has no answer for.
        """
        if not floors:
            return True
        checked = False
        for name, raw in moment["signals"].items():
            floor = floors.get(name)
            if floor is None or floor <= 0:
                continue          # this signal is its own filter; it cannot veto
            checked = True
            if raw >= floor:
                return True
        # Nothing had a floor to clear (keyword-only): trust the signal.
        return not checked

    @staticmethod
    def spread(moments: list[dict], k: int | None, per_video: int) -> list[dict]:
        """Stop one video from taking the whole page.

        Beyond the first `per_video` moments a video is demoted below every
        other video's, rather than removed: if there is nothing else to show,
        the page fills back up from the demoted ones in score order. Search
        should not hide an answer because of where it came from.
        """
        if per_video <= 0:
            return moments if k is None else moments[:k]

        kept, demoted, counts = [], [], {}
        for m in moments:
            n = counts.get(m["video_id"], 0)
            if n < per_video:
                counts[m["video_id"]] = n + 1
                kept.append(m)
            else:
                demoted.append(m)

        out = kept + demoted          # both halves already in score order
        return out if k is None else out[:k]

    @staticmethod
    def _moment(members: list[dict], peak: dict, decay: float) -> dict:
        """One assembled moment: its span, its score, and its evidence."""
        # Corroboration under a geometric discount, bounded by peak/(1 - decay).
        ordered = sorted((c["score"] for c in members), reverse=True)
        score = sum(s * decay ** i for i, s in enumerate(ordered))

        # The best each signal managed anywhere in the moment. A neighbouring
        # clip that matched on its caption is evidence for the moment even when
        # the peak clip matched on vision alone.
        signals: dict[str, float] = {}
        for c in members:
            for name, raw in c["signals"].items():
                if raw > signals.get(name, float("-inf")):
                    signals[name] = raw

        return {
            "video_id": peak["video_id"],
            "start_s": members[0]["start_s"],
            "end_s": members[-1]["end_s"],
            # Where playback should start: the strongest clip, not the edge of
            # the span. Seeking to the start of a 40s moment can drop the viewer
            # half a minute before the thing they searched for.
            "peak_s": peak["start_s"],
            "clip_id": peak["clip_id"],
            "score": score,
            "signals": signals,
            # What the interface should show. See MAX_RELEVANCE.
            "relevance": Searcher.confidence(signals),
            "locator": peak["locator"],
            # Every clip behind the moment, in time order, so the player can
            # mark them and the card can say how much evidence there is.
            "clips": [
                {"clip_id": c["clip_id"], "start_s": c["start_s"],
                 "end_s": c["end_s"], "score": c["score"],
                 "signals": c["signals"]}
                for c in members
            ],
        }

    # -- request handling --------------------------------------------------

    def handle(self, req: dict) -> dict:
        started = time.monotonic()

        query = (req.get("q") or "").strip()
        if not query:
            return {"ok": False, "error": "empty query"}

        scope_name = req.get("scope") or "corpus"
        video_id = req.get("video_id") or None
        if scope_name == "video" and not video_id:
            return {"ok": False, "error": "scope=video requires video_id"}
        if scope_name not in ("corpus", "video"):
            return {"ok": False, "error": f"unknown scope {scope_name!r} (want corpus or video)"}

        collection = req.get("collection") or "examples"
        if collection not in ("examples", "mine"):
            return {"ok": False,
                    "error": f"unknown collection {collection!r} (want examples or mine)"}
        scope = Scope(collection=collection, owner=req.get("owner"),
                      video_id=video_id if scope_name == "video" else None)

        weights = dict(DEFAULT_WEIGHTS)

        # Plan the query: what it is asking for, and how to phrase it for each
        # index. Synchronous and uncached -- a slow model makes search slow, and
        # a dead one leaves the raw query and the measured default weights.
        plan = None
        if as_bool(req.get("route"), True) and not req.get("weights"):
            plan = self.router.plan(query)
            if plan is not None:
                weights = plan.weights()

        override = req.get("weights")
        if isinstance(override, str):          # "visual:1,speech:0.3"
            for part in override.split(","):
                name, _, value = part.partition(":")
                if name.strip() and value.strip():
                    weights[name.strip()] = float(value)
        elif isinstance(override, dict):
            weights.update({str(a): float(b) for a, b in override.items()})

        k = int(req.get("k") or DEFAULT_K.get(scope_name, 8))
        k = max(1, min(k, 100))

        # Assembly is on by default and switchable, for the same reason signal
        # masks are: an ablation has to be able to measure the clip-level
        # ranking through the real search path, not a reimplementation of it.
        assemble = as_bool(req.get("assemble"), default=True)
        gap = as_float(req.get("moment_gap"), MOMENT_GAP)
        max_len = as_float(req.get("moment_max_len"), MOMENT_MAX_LEN)
        decay = as_float(req.get("moment_decay"), MOMENT_DECAY)
        floor = as_float(req.get("moment_floor"), MOMENT_FLOOR)
        per_video = int(as_float(req.get("moment_per_video"), MOMENT_PER_VIDEO))

        # Switchable so an ablation can measure what the floor costs, and so a
        # corpus whose similarities sit elsewhere can turn it off rather than
        # silently return nothing.
        floors = MIN_RELEVANCE if as_bool(req.get("min_relevance"), True) else None

        # Chosen after the weights, not before: a zero-weighted signal cannot
        # change the ranking, so retrieving it only costs latency -- but routing
        # changes which signals carry weight, and a speech query that never
        # retrieved speech would be routed to nothing. An explicit request still
        # runs whatever it asks for, so ablations measure what they intend to.
        requested = req.get("signals")
        signals = requested or [s for s in IMPLEMENTED if weights.get(s, 0.0) > 0.0]
        if isinstance(signals, str):
            signals = [s.strip() for s in signals.split(",") if s.strip()]
        for s in signals:
            if s not in KNOWN:
                return {"ok": False, "error": f"unknown signal {s!r} (known: {', '.join(KNOWN)})"}
            if s not in IMPLEMENTED:
                return {"ok": False,
                        "error": f"signal {s!r} is not implemented yet "
                                 f"(available: {', '.join(IMPLEMENTED)})"}

        # Each index gets its own phrasing of the query. Without a plan they all
        # get the raw query, which is exactly the previous behaviour.
        retrieve = {"visual": self.visual_candidates, "caption": self.caption_candidates,
                    "speech": self.speech_candidates, "keyword": self.keyword_candidates}
        lists = {}
        for s in signals:
            text = plan.text_for(s, query) if plan else query
            lists[s] = retrieve[s](text, scope)

        # Fusion runs over the whole candidate pool, not the top k: a moment
        # built from a truncated list would be cut off at the page boundary.
        fused = self.fuse(lists, None, weights)
        results = (self.assemble(fused, k, gap, max_len, decay, floor, per_video, floors)
                   if assemble else fused[:k])

        return {
            "ok": True,
            "query": query,
            "scope": scope_name,
            "collection": collection,
            "video_id": video_id,
            "signals": signals,
            "k": k,
            # Recall@1 is meaningless without the distractor count, so every
            # response carries what was actually searched.
            "corpus": self.corpus_size(scope),
            "coverage": self.coverage(scope),
            "weights": {s: weights.get(s, 1.0) for s in signals},
            "plan": plan.as_json() if plan else None,
            "assemble": assemble,
            "moment": ({"gap": gap, "max_len": max_len, "decay": decay,
                        "floor": floor, "per_video": per_video,
                        "min_relevance": bool(floors)} if assemble else None),
            "results": results,
            "took_ms": round((time.monotonic() - started) * 1000, 2),
        }


class RemoteSearcher:
    """Answers the same protocol by running each query on Kaggle.

    Exists because the deployment has no RAM for the encoders -- not even the
    two that search needs. It is slow: one Kaggle kernel per query, minutes
    rather than milliseconds. That is the cost of the free tier, and it is
    visible to the caller only as latency.

    Only one query runs at a time. Kaggle allows very few concurrent kernels
    per account, and a second push while the first is running gets queued
    behind it anyway -- the lock makes the wait explicit instead of turning it
    into a pile of pending kernels.
    """

    def __init__(self):
        import threading

        from python.lib import remote

        self._remote = remote
        self._lock = threading.Lock()

    def handle(self, req: dict) -> dict:
        from python.lib.config import DATABASE_URL, GEMINI_API_KEY
        from python.lib.remote import Job, RemoteError

        job = Job(kind="search", gpu=False, payload={
            "request": req,
            # The kernel reads these from Kaggle Secrets when they are set; the
            # values here are the fallback for an account not yet configured.
            "database_url": DATABASE_URL,
            "gemini_api_key": GEMINI_API_KEY,
        })
        # The serve loop is already one request at a time and Go serialises on
        # top of that, so this lock is belt and braces -- but a second kernel
        # pushed while the first runs would sit in Kaggle's queue anyway, and
        # waiting here is cheaper than waiting there.
        with self._lock:
            try:
                return self._remote.run(job)
            except RemoteError as exc:
                return {"ok": False, "error": str(exc)}


def serve(searcher: Searcher) -> int:
    """One JSON request per line on stdin, one response per line on stdout."""
    # Warm whichever encoders the default weights actually use, so the first
    # real query does not pay the load. Signals weighted to zero stay unloaded.
    # A remote searcher holds no encoders: there is nothing here to warm.
    for signal in [] if isinstance(searcher, RemoteSearcher) else IMPLEMENTED:
        if DEFAULT_WEIGHTS.get(signal, 0.0) <= 0.0:
            continue
        if signal == "visual":
            searcher.siglip
        elif signal in ("caption", "speech"):
            searcher.bge

    print(json.dumps({"event": "ready", "dim": VISUAL_DIM}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = searcher.handle(json.loads(line))
        except Exception as exc:                     # never let the worker die
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Search the video index.")
    ap.add_argument("--serve", action="store_true", help="run as a stdin/stdout worker")
    ap.add_argument("--query", help="one-shot search, for debugging")
    ap.add_argument("--scope", default="corpus", choices=["corpus", "video"])
    ap.add_argument("--collection", default="examples", choices=["examples", "mine"])
    ap.add_argument("--video-id")
    ap.add_argument("-k", type=int, default=0, help="0 uses the per-scope default")
    ap.add_argument("--no-assemble", action="store_true",
                    help="return raw clips instead of assembled moments")
    ap.add_argument("--no-route", action="store_true",
                    help="skip query planning: raw query, measured default weights")
    ap.add_argument("--remote", action="store_true",
                    help="run each query on Kaggle instead of loading models here")
    args = ap.parse_args()

    if not args.serve and not args.query:
        ap.error("pass --serve or --query")

    searcher = RemoteSearcher() if args.remote else Searcher()
    if args.serve:
        return serve(searcher)

    result = searcher.handle({"q": args.query, "scope": args.scope,
                              "collection": args.collection,
                              "video_id": args.video_id, "k": args.k,
                              "assemble": not args.no_assemble,
                              "route": not args.no_route})
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
