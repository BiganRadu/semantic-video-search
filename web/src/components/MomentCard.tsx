import { Link } from "react-router-dom";
import type { Moment } from "../api/types";
import { mmss, watchUrl } from "../api/time";
import { ExternalIcon } from "./Icons";
import Thumbnail from "./Thumbnail";

/**
 * One ranked moment, with the evidence for it. The per-signal scores are shown
 * because a result you cannot explain is a result you cannot debug — and this
 * is the panel that made near-duplicate flooding obvious.
 */
export default function MomentCard({
  moment, rank, query,
}: {
  moment: Moment;
  rank: number;
  query?: string;
}) {
  // Seek to the strongest clip, not the edge of the span: the start of a 20s
  // moment can be a full clip before the thing that was searched for.
  const seek = moment.peak_s ?? moment.start_s;
  const url = watchUrl(moment.locator, seek);
  const best = Math.max(0, ...Object.values(moment.signals));
  const to = `/video/${encodeURIComponent(moment.video_id)}?t=${seek}${
    query ? `&q=${encodeURIComponent(query)}` : ""
  }`;
  const span = Math.round(moment.end_s - moment.start_s);
  const parts = moment.clips?.length ?? 1;

  return (
    <div className="card interactive moment">
      <span className={`rank${rank <= 3 ? " top" : ""}`}>{rank}</span>

      {/*
        The video's poster frame, not this moment's frame — YouTube only serves
        stills at fixed positions, and getting one at an arbitrary timestamp
        would mean decoding the video. So it identifies which video a result is
        from; the timecode says where in it.
      */}
      <Link to={to} className="moment-thumb" aria-hidden="true" tabIndex={-1}>
        <Thumbnail locator={moment.locator} alt="" seed={moment.video_id} />
      </Link>

      <div className="grow" style={{ minWidth: 0 }}>
        <div className="row" style={{ gap: 9 }}>
          <span className="timecode">
            {mmss(moment.start_s)}–{mmss(moment.end_s)}
          </span>
          <span className="chip span" title={
            parts > 1
              ? `${parts} adjacent clips matched and were merged into one moment`
              : "one clip matched"
          }>
            {span}s{parts > 1 && ` · ${parts} clips`}
          </span>
          <Link to={to} className="truncate dim" style={{ fontSize: 13.5 }}>
            {moment.video_id}
          </Link>
        </div>

        <div className="chips">
          {Object.entries(moment.signals).map(([name, score]) => (
            <span className="chip signal" key={name}>
              {name} <b>{score.toFixed(3)}</b>
            </span>
          ))}
        </div>

        {/* relative strength of the best matching signal, at a glance */}
        <div className="meter" aria-hidden="true">
          <span style={{ width: `${Math.min(100, best * 100).toFixed(1)}%` }} />
        </div>
      </div>

      <div className="row" style={{ gap: 4 }}>
        <Link to={to}>
          <button className="ghost" title="open in the player">open</button>
        </Link>
        {url && (
          <a href={url} target="_blank" rel="noopener noreferrer">
            <button className="ghost" title="watch at the source">
              <ExternalIcon />
            </button>
          </a>
        )}
      </div>
    </div>
  );
}
