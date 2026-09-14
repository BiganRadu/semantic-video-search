"""Write one indexed video into the database.

Lives here rather than in scripts/ because the Kaggle kernel needs it: a remote
index writes its own rows -- a 40-minute video is ~15k frame embeddings, which
is not something to send back through a notebook log -- and the kernel is given
python/, not the dev scripts.

The search path must never import this. It writes, and search only reads.
"""
from __future__ import annotations

import json

from python.lib.config import PIPELINE_VERSION
from python.lib.models import vector_literal


def save(conn, video_id: str, locator: dict, payload: dict,
         source: str = "corpus", title: str | None = None) -> None:
    """Write one indexed video in a single transaction.

    The video is the unit of atomicity: a half-written video is never visible,
    and rerunning replaces it wholesale.

    `owner` is left NULL deliberately: a bulk-loaded video belongs to no session
    and no account, which is what makes it visible in the shared corpus rather
    than only to whoever loaded it.
    """
    v = payload["video"]
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO videos (id, locator, source, title, duration_s, fps, state,
                                indexed_at, pipeline)
            VALUES (%s, %s, %s, %s, %s, %s, 'ready', now(), %s)
            ON CONFLICT (id) DO UPDATE SET
                locator=EXCLUDED.locator, source=EXCLUDED.source,
                title=EXCLUDED.title, duration_s=EXCLUDED.duration_s,
                fps=EXCLUDED.fps, state='ready', indexed_at=now(),
                pipeline=EXCLUDED.pipeline, last_error=NULL, progress=NULL
            """,
            (video_id, json.dumps(locator), source, title,
             v["duration_s"], v["fps"], PIPELINE_VERSION),
        )
        cur.execute("DELETE FROM clips WHERE video_id = %s", (video_id,))
        cur.execute("DELETE FROM transcript_segments WHERE video_id = %s", (video_id,))

        lang = payload.get("language")
        for seg in payload.get("transcript", []):
            cur.execute(
                """INSERT INTO transcript_segments
                       (video_id, start_s, end_s, text, lang, avg_logprob, no_speech_prob)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (video_id, seg["start_s"], seg["end_s"], seg["text"], lang,
                 seg["avg_logprob"], seg["no_speech_prob"]),
            )

        model = payload.get("caption_model")
        for clip in payload["clips"]:
            cur.execute(
                """INSERT INTO clips (video_id, idx, start_s, end_s, static_score,
                                      novelty, speech, lang, caption, people, objects,
                                      actions, setting, tags_text, caption_model,
                                      caption_state)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (video_id, clip["idx"], clip["start_s"], clip["end_s"],
                 clip["static_score"], clip["novelty"], clip["speech"], clip["lang"],
                 clip.get("caption"),
                 json.dumps(clip.get("people")) if clip.get("people") else None,
                 json.dumps(clip.get("objects")) if clip.get("objects") else None,
                 json.dumps(clip.get("actions")) if clip.get("actions") else None,
                 clip.get("setting"), clip.get("tags_text"), model,
                 "ok" if clip.get("caption") else "empty"),
            )
            clip_id = cur.fetchone()[0]

            if any(clip.get(k) for k in ("visual", "speech_vec", "caption_vec")):
                cur.execute(
                    """INSERT INTO clip_vectors (clip_id, visual, speech_vec, caption_vec)
                       VALUES (%s, %s::vector, %s::vector, %s::vector)""",
                    (clip_id,
                     vector_literal(decode(clip["visual"])) if clip.get("visual") else None,
                     vector_literal(decode(clip["speech_vec"])) if clip.get("speech_vec") else None,
                     vector_literal(decode(clip["caption_vec"])) if clip.get("caption_vec") else None),
                )
            for f in clip["frames"]:
                cur.execute(
                    """INSERT INTO clip_frames (clip_id, frame_idx, t_s, embedding)
                       VALUES (%s,%s,%s,%s::halfvec)""",
                    (clip_id, f["idx"], f["t_s"], vector_literal(decode(f["embedding"]))),
                )


def decode(b64: str):
    import base64
    import numpy as np
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32)
