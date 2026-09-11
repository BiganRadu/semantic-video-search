import { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { ClipDetail, Moment, VideoDetailResponse } from "../api/types";
import { embedUrl, mmss } from "../api/time";

export default function Player() {
  const { id = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const at = Number(params.get("t") ?? 0);
  const query = params.get("q") ?? "";

  const [video, setVideo] = useState<VideoDetailResponse | null>(null);
  const [matches, setMatches] = useState<Moment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState(query);
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    api.video(id).then(setVideo).catch((e) => setError(String(e.message ?? e)));
  }, [id]);

  // Other matches inside this video: the same search, scoped to one video.
  //
  // The collection comes from the server, which derives it from ownership. It
  // used to be guessed from `video.source`, but source says where a video came
  // from, not whose it is -- so a video whose ownership changed after indexing
  // was searched in the wrong corpus and silently returned nothing.
  useEffect(() => {
    if (!query || !video) {
      setMatches([]);
      return;
    }
    let cancelled = false;
    setSearching(true);
    api
      .search({ q: query, scope: "video", videoId: id,
                collection: video.collection ?? "examples" })
      .then((r) => !cancelled && setMatches(r.results))
      .catch(() => !cancelled && setMatches([]))
      .finally(() => !cancelled && setSearching(false));
    return () => { cancelled = true; };
  }, [id, query, video]);

  const current = useMemo(
    () => video?.clips.find((c) => at >= c.start_s && at < c.end_s) ?? video?.clips[0],
    [video, at],
  );

  if (error) return <div className="notice error">{error}</div>;
  if (!video) return <div className="muted">loading…</div>;

  const embed = embedUrl(video.locator, at);

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="page-head">
        <h1 className="truncate">{video.title || video.id}</h1>
        <div className="chips" style={{ marginTop: 8 }}>
          <span className="chip">{mmss(video.duration_s)}</span>
          <span className="chip">{video.clips.length} clips</span>
          <span className="chip">{video.source}</span>
          {video.language && <span className="chip">{video.language}</span>}
          {video.pipeline && <span className="chip">{video.pipeline}</span>}
        </div>
      </div>

      {embed ? (
        <iframe
          className="embed"
          src={embed}
          title={video.id}
          allow="accelerometer; encrypted-media; picture-in-picture"
          allowFullScreen
        />
      ) : video.locator.kind === "http" ? (
        <video className="embed" src={`${video.locator.url}#t=${at}`} controls />
      ) : (
        <div className="notice">This video cannot be played from here.</div>
      )}

      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          setParams({ t: String(at), q: draft.trim() });
        }}
      >
        <input
          className="grow"
          value={draft}
          placeholder="find a moment inside this video"
          onChange={(e) => setDraft(e.target.value)}
        />
        <button type="submit">Find</button>
      </form>

      <Timeline
        duration={video.duration_s}
        at={at}
        matches={matches}
        onPick={(t) => setParams(query ? { t: String(t), q: query } : { t: String(t) })}
      />

      {query && (
        <Matches
          matches={matches}
          searching={searching}
          at={at}
          onPick={(t) => setParams({ t: String(t), q: query })}
          clips={video.clips}
        />
      )}

      {current && <ClipPanel clip={current} />}

      {video.transcript.length > 0 && (
        <div>
          <div className="muted" style={{ marginBottom: 6 }}>transcript</div>
          <div className="stack">
            {video.transcript.map((seg, i) => (
              <button
                key={i}
                className="card interactive"
                style={{ textAlign: "left", display: "block", width: "100%" }}
                onClick={() => setParams(query ? { t: String(seg.start_s), q: query } : { t: String(seg.start_s) })}
              >
                <span className="timecode">{mmss(seg.start_s)}</span>{" "}
                <span>{seg.text}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <Link to={video.collection === "mine" ? "/mine" : "/"} className="muted">
        ← back to {video.collection === "mine" ? "your videos" : "example videos"}
      </Link>
    </div>
  );
}

/**
 * Matching moments on the video's timeline.
 *
 * A mark covers the moment's whole span, not just the clip it started on, and
 * its opacity tracks rank — the strongest match is the one that stands out.
 * Marking every hit identically is what made a search look like it had done
 * nothing: the bar simply filled up.
 */
function Timeline({
  duration, at, matches, onPick,
}: {
  duration: number;
  at: number;
  matches: Moment[];
  onPick: (t: number) => void;
}) {
  // Opacity tracks raw relevance, not the fusion score: inside one video the
  // fusion scores are all within a few percent of each other, so a mark drawn
  // from them is a mark that says nothing.
  return (
    <div>
      <div className="timeline">
        <div className="track" />
        {matches.map((m, i) => (
          <button
            key={`${m.video_id}-${m.start_s}`}
            className="mark"
            style={{
              left: `${(m.start_s / duration) * 100}%`,
              width: `${Math.max(0.8, ((m.end_s - m.start_s) / duration) * 100)}%`,
              opacity: 0.3 + 0.7 * (m.relevance ?? 1),
            }}
            title={`#${i + 1} · ${mmss(m.start_s)}–${mmss(m.end_s)}`}
            onClick={() => onPick(m.peak_s ?? m.start_s)}
          />
        ))}
        <span className="playhead" style={{ left: `${(at / duration) * 100}%` }} />
      </div>
    </div>
  );
}

/**
 * The ranked moments themselves.
 *
 * The player used to compute this list and render it only as tick marks, which
 * is why searching inside a video looked like it did nothing. An empty result
 * is now stated rather than left to look like a failure — with a relevance
 * floor in place, "no moment in this video matches" is a real answer.
 */
function Matches({
  matches, searching, at, onPick, clips,
}: {
  matches: Moment[];
  searching: boolean;
  at: number;
  onPick: (t: number) => void;
  clips: ClipDetail[];
}) {
  if (searching) return <div className="skeleton-row" aria-label="searching" />;
  if (!matches.length) {
    return (
      <div className="notice subtle">
        Nothing in this video matches that. Try different words — or search the
        whole library from the search page.
      </div>
    );
  }
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="muted small">
        {matches.length} matching {matches.length === 1 ? "moment" : "moments"} in this video
      </div>
      {matches.map((m, i) => {
        const seek = m.peak_s ?? m.start_s;
        const active = at >= m.start_s && at < m.end_s;
        const caption = clips.find((c) => c.start_s === seek)?.caption;
        return (
          <button
            key={`${m.start_s}-${m.end_s}`}
            className={`card interactive moment${active ? " active" : ""}`}
            onClick={() => onPick(seek)}
          >
            <span className={`rank${i < 3 ? " top" : ""}`}>{i + 1}</span>
            <div className="grow" style={{ minWidth: 0, textAlign: "left" }}>
              <div className="row" style={{ gap: 9 }}>
                <span className="timecode">{mmss(m.start_s)}–{mmss(m.end_s)}</span>
                {m.relevance !== undefined && (
                  <span className="chip signal">
                    match <b>{Math.round(m.relevance * 100)}%</b>
                  </span>
                )}
                <span className="chips">
                  {Object.entries(m.signals).map(([name, score]) => (
                    <span className="chip signal" key={name}>
                      {name} <b>{score.toFixed(2)}</b>
                    </span>
                  ))}
                </span>
              </div>
              {caption && <div className="caption-line truncate-2">{caption}</div>}
            </div>
          </button>
        );
      })}
    </div>
  );
}

/** Why this moment matched, in the model's own words. */
function ClipPanel({ clip }: { clip: ClipDetail }) {
  return (
    <div className="card">
      <span className="timecode">{mmss(clip.start_s)}–{mmss(clip.end_s)}</span>
      {clip.caption ? (
        <div className="caption-line">{clip.caption}</div>
      ) : (
        <div className="caption-line muted">No caption indexed for this clip yet.</div>
      )}
      {(clip.objects?.length || clip.actions?.length || clip.setting) && (
        <div className="chips">
          {clip.setting && <span className="chip">{clip.setting}</span>}
          {clip.actions?.map((a) => <span className="chip" key={`a-${a}`}>{a}</span>)}
          {clip.objects?.map((o) => <span className="chip" key={`o-${o}`}>{o}</span>)}
        </div>
      )}
      {clip.speech && <div className="speech-line">“{clip.speech}”</div>}
    </div>
  );
}
