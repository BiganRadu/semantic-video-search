import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { IndexJob } from "../api/types";

/**
 * What this visitor is waiting on, parked in the bottom-left corner.
 *
 * Indexing runs one video at a time and takes about a minute of GPU per minute
 * of video, so "I submitted it, is anything happening?" is the question the UI
 * most needs to answer. The add page answers it only while you stay on the add
 * page; this follows you around the app.
 *
 * It hides itself when there is nothing to report, so the corner is empty in
 * the normal case rather than showing a permanently-zero widget.
 */
export default function JobsPanel() {
  const [jobs, setJobs] = useState<IndexJob[]>([]);
  const [open, setOpen] = useState(false);
  const [dismissed, setDismissed] = useState<string[]>([]);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    let alive = true;

    const tick = async () => {
      try {
        const r = await api.jobs();
        if (!alive) return;
        setJobs(r.jobs);
        // Poll briskly while something is moving, slowly when it is not: an
        // idle tab should not ask a question every two seconds forever.
        const busy = r.jobs.some((j) => j.state === "queued" || j.state === "indexing");
        timer.current = window.setTimeout(tick, busy ? 2000 : 15000);
      } catch {
        if (alive) timer.current = window.setTimeout(tick, 15000);
      }
    };

    tick();
    return () => {
      alive = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, []);

  const shown = jobs.filter((j) => !dismissed.includes(j.id));
  const active = shown.filter((j) => j.state === "queued" || j.state === "indexing");

  // Open on its own when work starts, so a submitted video is visibly under
  // way without the panel having to be found first.
  useEffect(() => {
    if (active.length > 0) setOpen(true);
  }, [active.length > 0]);

  if (shown.length === 0) return null;

  // Say what is actually happening. Only one video indexes at a time, so
  // lumping queued jobs in as "indexing" would claim parallelism that the
  // server deliberately does not have.
  const running = shown.filter((j) => j.state === "indexing").length;
  const waiting = shown.filter((j) => j.state === "queued").length;
  const summary = running > 0
    ? (waiting > 0 ? `indexing · ${waiting} waiting` : "indexing")
    : waiting > 0
      ? `${waiting} waiting`
      : `${shown.length} finished`;

  return (
    <div className="jobs-dock">
      {open && (
        <div className="jobs-panel card">
          <div className="row between jobs-head">
            <strong>Indexing</strong>
            <button className="ghost" onClick={() => setOpen(false)} title="collapse">
              ▾
            </button>
          </div>
          <div className="stack" style={{ gap: 8 }}>
            {shown.map((j) => (
              <JobRow key={j.id} job={j}
                      onDismiss={() => setDismissed((d) => [...d, j.id])} />
            ))}
          </div>
        </div>
      )}

      <button className={`jobs-toggle${active.length ? " busy" : ""}`}
              onClick={() => setOpen((o) => !o)}>
        {active.length > 0 && <span className="spinner" aria-hidden="true" />}
        {summary}
      </button>
    </div>
  );
}

function JobRow({ job, onDismiss }: { job: IndexJob; onDismiss: () => void }) {
  const name = job.title || job.id;

  return (
    <div className="job-row">
      <div className="row between" style={{ gap: 8 }}>
        {job.state === "done" ? (
          <Link to={`/video/${encodeURIComponent(job.id)}`} className="truncate job-name">
            {name}
          </Link>
        ) : (
          <span className="truncate job-name" title={name}>{name}</span>
        )}
        {(job.state === "done" || job.state === "failed") && (
          <button className="ghost" onClick={onDismiss} title="dismiss">×</button>
        )}
      </div>

      {/*
        The state, and nothing else. The pipeline's stages -- caption, embed,
        transcribe, embed_speech -- are how the indexer is built, not something
        anyone waiting on a video needs to follow. A failure is the exception:
        there the reason is the whole point.
      */}
      <div className="row" style={{ gap: 6 }}>
        <span className={`job-state ${job.state}`}>{label(job)}</span>
        {job.state === "failed" && job.error && (
          <span className="muted small truncate" title={job.error}>{job.error}</span>
        )}
      </div>
    </div>
  );
}

function label(job: IndexJob): string {
  if (job.state !== "queued") return job.state;
  // How many are in front is queue position, not pipeline detail: it is the
  // difference between "soon" and "not for an hour".
  const ahead = (job.position ?? 1) - 1;
  return ahead > 0 ? `waiting · ${ahead} ahead` : "waiting";
}
