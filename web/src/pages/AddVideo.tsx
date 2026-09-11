import { useState } from "react";
import { Link } from "react-router-dom";
import { useConfig } from "../config";

interface ProgressEvent {
  event: "progress" | "done" | "error";
  stage?: string;
  done?: number;
  total?: number;
  message?: string;
  video_id?: string;
  clips?: number;
  duration_s?: number;
}

export default function AddVideo() {
  const { indexing_enabled } = useConfig();
  const [url, setUrl] = useState("");
  const [lines, setLines] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  if (!indexing_enabled) {
    return (
      <div className="notice">
        <b>Indexing is off on this deployment.</b>
        <br />
        Building an index needs a GPU, so it runs locally. This instance serves an index
        that was built elsewhere.
      </div>
    );
  }

  const say = (line: string) => setLines((current) => [...current, line]);

  // Progress arrives as server-sent events while index.py runs, so a video that
  // takes minutes shows what stage it is on rather than just hanging.
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setDone(false);
    setLines([]);
    try {
      const res = await fetch("/api/videos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      if (!res.body) {
        say(`error: ${(await res.json().catch(() => ({}))).error ?? "no response stream"}`);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let cut: number;
        while ((cut = buffer.indexOf("\n\n")) >= 0) {
          const chunk = buffer.slice(0, cut);
          buffer = buffer.slice(cut + 2);
          if (!chunk.startsWith("data: ")) continue;
          const ev: ProgressEvent = JSON.parse(chunk.slice(6));
          if (ev.event === "progress") {
            // The download stage counts percent, every other stage counts items.
            if (ev.stage === "download" && ev.total === 100) {
              setLines((c) => {
                const line = `download ${ev.done}%`;
                // one line that updates, not one line per progress tick
                return c.length && c[c.length - 1].startsWith("download ")
                  ? [...c.slice(0, -1), line] : [...c, line];
              });
            } else {
              say(ev.total && ev.total > 1 ? `${ev.stage} ${ev.done}/${ev.total}` : `${ev.stage}`);
            }
          } else if (ev.event === "done") {
            say(`done — ${ev.clips} clips, ${Math.round(ev.duration_s ?? 0)}s`);
            setDone(true);
          } else if (ev.event === "error") {
            say(`error: ${ev.message}`);
          }
        }
      }
    } catch (err) {
      say(`error: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack" style={{ gap: 14 }}>
      <div className="page-head">
        <h1>Add a video</h1>
        <div className="sub">
          Paste a YouTube link or a direct video URL. It is fetched, indexed, then
          deleted — only a pointer to the original is stored, never the video bytes.
          It lands in <b>Your videos</b>.
        </div>
      </div>

      <form className="row" onSubmit={submit}>
        <input
          className="grow"
          type="url"
          required
          placeholder="https://www.youtube.com/watch?v=… or https://example.com/clip.mp4"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "indexing…" : "index"}
        </button>
      </form>

      <div className="muted">
        Fetched with <code>yt-dlp</code>, so most video sites work, not only YouTube.
        Indexing takes roughly a minute of GPU time per minute of video.
      </div>

      {lines.length > 0 && <pre className="log">{lines.join("\n")}</pre>}
      {done && (
        <Link to="/mine">
          <button className="primary">See it in your videos →</button>
        </Link>
      )}
    </div>
  );
}
