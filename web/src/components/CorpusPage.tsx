import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { Collection, SearchResponse, VideoSummary } from "../api/types";
import { hours, mmss } from "../api/time";
import MomentCard from "./MomentCard";
import Thumbnail from "./Thumbnail";
import SearchBar from "./SearchBar";
import { ClockIcon, FilmIcon, TrashIcon } from "./Icons";

/**
 * A corpus page: a search bar over one collection, plus the videos in it.
 *
 * Both pages are the same surface — searching your own uploads should not feel
 * like a different product from searching the examples — so the differences are
 * props, not a second implementation.
 */
export default function CorpusPage({
  collection, title, blurb, canManage, onAdd, emptyState,
}: {
  collection: Collection;
  title: string;
  blurb: string;
  canManage?: boolean;
  onAdd?: React.ReactNode;
  emptyState: React.ReactNode;
}) {
  const [params, setParams] = useSearchParams();
  const query = params.get("q") ?? "";

  const [videos, setVideos] = useState<VideoSummary[] | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadVideos = () =>
    api.videos(collection)
      .then((r) => setVideos(r.videos))
      .catch((e) => setError(String(e.message ?? e)));

  useEffect(() => {
    setVideos(null);
    setResult(null);
    loadVideos();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collection]);

  useEffect(() => {
    if (!query) {
      setResult(null);
      return;
    }
    let cancelled = false;
    setBusy(true);
    setError(null);
    api.search({ q: query, collection })
      .then((r) => !cancelled && setResult(r))
      .catch((e) => !cancelled && setError(String(e.message ?? e)))
      .finally(() => !cancelled && setBusy(false));
    return () => { cancelled = true; };
  }, [query, collection]);

  const remove = async (id: string) => {
    if (!confirm(`Remove "${id}" from your index?\n\nThe video itself is untouched.`)) return;
    try {
      await api.deleteVideo(id);
      setVideos((current) => current?.filter((v) => v.id !== id) ?? null);
      if (query) setParams({ q: query });
    } catch (e) {
      setError(String((e as Error).message));
    }
  };

  // Total runtime, not clip count: a clip is a 10s indexing window, which is
  // an implementation detail of the pipeline and not a unit anyone searches in.
  const seconds = videos?.reduce((sum, v) => sum + (v.duration_s ?? 0), 0) ?? 0;

  return (
    <div className="stack" style={{ gap: 20 }}>
      <div className="page-head between">
        <div>
          <h1>{title}</h1>
          <div className="sub">{blurb}</div>
        </div>
        {onAdd}
      </div>

      <SearchBar
        value={query}
        busy={busy}
        onSubmit={(q) => setParams(q ? { q } : {})}
        placeholder={
          collection === "mine"
            ? "search across the videos you added"
            : "a person in a red jacket carries a box out the back door"
        }
      />

      {error && <div className="notice error">{error}</div>}

      {query ? (
        <Results result={result} busy={busy} query={query} />
      ) : videos === null ? (
        <div className="stack">
          {[0, 1, 2].map((i) => <div className="skeleton" key={i} />)}
        </div>
      ) : videos.length === 0 ? (
        emptyState
      ) : (
        <>
          <div className="results-meta">
            <span><FilmIcon /></span>
            <span>{videos.length} videos · {hours(seconds)} indexed</span>
          </div>
          <div className="grid">
            {videos.map((v) => (
              <VideoCard key={v.id} video={v} onDelete={canManage ? remove : undefined} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function Results({
  result, busy, query,
}: {
  result: SearchResponse | null;
  busy: boolean;
  query: string;
}) {
  if (busy && !result) {
    return (
      <div className="stack">
        {[0, 1, 2, 3, 4].map((i) => <div className="skeleton" key={i} />)}
      </div>
    );
  }
  if (!result) return null;

  if (result.results.length === 0) {
    return (
      <div className="notice">
        No moments matched <b>“{query}”</b> in this corpus.
        <div className="muted" style={{ marginTop: 6 }}>
          Searched all {result.corpus.videos} videos in this corpus.
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="results-meta">
        <span>{result.results.length} moments</span>
        <span>·</span>
        <span>across {result.corpus.videos} videos</span>
        <span>·</span>
        <span><ClockIcon /> {result.took_ms.toFixed(0)} ms</span>
        <span>·</span>
        <span>{result.signals.join(" + ")}</span>
        {result.plan && (
          <>
            <span>·</span>
            <span
              className="chip plan"
              title={
                "The query was classified to pick the weights, and rephrased " +
                "for each index:\n\n" +
                Object.entries(result.plan.queries)
                  .map(([k, v]) => `${k}: ${v}`).join("\n")
              }
            >
              {result.plan.class} query
            </span>
          </>
        )}
      </div>
      <div className="stack">
        {result.results.map((m, i) => (
          <MomentCard key={m.clip_id} moment={m} rank={i + 1} query={query} />
        ))}
      </div>
    </>
  );
}

function VideoCard({
  video, onDelete,
}: {
  video: VideoSummary;
  onDelete?: (id: string) => void;
}) {
  const title = video.title || video.id;
  return (
    <div className="card interactive video-card">
      {/* The whole still is the link — a 260px-wide target beats a line of text. */}
      <Link to={`/video/${encodeURIComponent(video.id)}`} aria-label={title}>
        <Thumbnail locator={video.locator} alt="" seed={video.id} />
        <span className="duration">{mmss(video.duration_s)}</span>
      </Link>

      <div className="video-body">
        <Link to={`/video/${encodeURIComponent(video.id)}`} className="video-title" title={title}>
          {title}
        </Link>
        <div className="row between" style={{ marginTop: 8 }}>
          <div className="chips">
            <span className="chip">{video.locator.kind}</span>
            {video.state !== "ready" && <span className="chip">{video.state}</span>}
          </div>
          {onDelete && (
            <button className="ghost danger" onClick={() => onDelete(video.id)}
                    title="remove from your index">
              <TrashIcon />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}


