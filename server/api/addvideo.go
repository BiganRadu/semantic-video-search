package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"path/filepath"
	"strings"
	"time"

	"videosearch/server/pyproc"
	"videosearch/server/store"
)

// Indexing runs at roughly a minute of GPU per minute of video and the source
// duration is already capped, so this bounds a hung yt-dlp or a wedged model
// load rather than a slow video.
const indexTimeout = 2 * time.Hour

type addVideoRequest struct {
	URL   string `json:"url"`
	Title string `json:"title"`
}

// apiAddVideo hands a link to index.py and writes back what comes out.
//
// Go does not fetch the video. index.py resolves the link with yt-dlp,
// downloads it, indexes it and deletes the bytes, then returns the clips along
// with the locator pointing at wherever the video already lives. Doing it here
// instead would mean two places that know how to turn a link into a playable
// stream -- and the one without yt-dlp would be the one deciding what gets
// stored forever.
//
// Progress streams back as server-sent events, so a video that takes minutes
// shows what stage it is on rather than just hanging.
func (s *Server) apiAddVideo(w http.ResponseWriter, r *http.Request) {
	var req addVideoRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<10)).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "expected a JSON body with a url"})
		return
	}

	// Only the obvious rejections happen here, so a bad link fails before a
	// subprocess is started. yt-dlp checks the same things again on its side.
	name, err := slugFor(req.URL)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	// Namespace the id by owner: two visitors adding the same URL must not
	// collide on one row, where the second would silently take over the first's.
	owner := ownerKey(r)
	videoID := fmt.Sprintf("u_%s_%s", shortHash(owner), name)

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "streaming unsupported"})
		return
	}

	// A video kept by an account is kept until its owner deletes it; one added
	// without an account rides on the anonymous session and lapses with it.
	// Decided here, while the request is still in hand -- the job outlives it.
	var expires *time.Time
	if userFrom(r) == nil {
		lapse := time.Now().Add(videoTTL)
		expires = &lapse
	}
	title := req.Title
	url := req.URL

	// Detached from this request on purpose. r.Context() dies when the browser
	// does, and everything below -- the subprocess, and the save that makes its
	// work permanent -- has to survive that.
	bg := context.WithoutCancel(r.Context())

	// Until the pipeline resolves the real title, the submitted link is a
	// better label than the generated row id.
	label := title
	if label == "" {
		label = url
	}

	j, started := s.jobs.startOrAttach(videoID, owner, label, func(j *job) {
		emit := j.emit
		ctx, cancel := context.WithTimeout(bg, indexTimeout)
		defer cancel()

		// Claim the row first, so an index that dies mid-flight leaves a trace.
		if err := s.store.MarkIndexing(ctx, videoID, "user", owner, expires); err != nil {
			emit(map[string]any{"event": "error", "message": err.Error()})
			return
		}

		result, err := s.indexer.Run(ctx, url, videoID, func(ev pyproc.Event) { emit(ev) })
		if err != nil {
			s.markFailed(ctx, videoID, err)
			emit(map[string]any{"event": "error", "message": err.Error()})
			return
		}

		result.Source = "user"
		// index.py reports the video's own title; a title typed on the form
		// wins, because someone who bothered to name it meant it.
		if title != "" {
			result.Title = title
		}
		// Now the status panel can name the video instead of showing a slug.
		j.setTitle(result.Title)
		result.Owner = &owner
		result.ExpiresAt = expires

		emit(pyproc.Event{Event: "progress", Stage: "save"})
		if err := s.store.Save(ctx, result); err != nil {
			s.markFailed(ctx, videoID, err)
			emit(map[string]any{"event": "error", "message": err.Error()})
			return
		}

		done := map[string]any{
			"event": "done", "video_id": videoID,
			"clips": len(result.Clips), "duration_s": result.DurationS,
		}
		if expires != nil {
			done["expires_at"] = expires.Format(time.RFC3339)
		}
		emit(done)
	})

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	send := func(payload any) {
		b, _ := json.Marshal(payload)
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	if !started {
		// A refresh mid-index lands here: say so, then replay from the start so
		// the page shows the stage it is actually on rather than restarting.
		send(map[string]any{"event": "progress", "stage": "already queued"})
	}

	history, ch := j.attach()
	for _, ev := range history {
		send(ev)
	}
	if ch == nil {
		return // finished before we attached; history was the whole story
	}
	defer j.detach(ch)

	seen := len(history)
	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				// Job ended. Anything dropped while we were slow is still in
				// history, so the client always sees the terminal event.
				for _, ev := range j.since(seen) {
					send(ev)
				}
				return
			}
			seen++
			send(ev)
		case <-r.Context().Done():
			return // the viewer left; the job carries on without them
		}
	}
}

// markFailed records a failure without letting a dead context hide it: the
// usual cause of the failure is the context, and reporting it needs a live one.
func (s *Server) markFailed(ctx context.Context, videoID string, cause error) {
	if ctx.Err() != nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := s.store.MarkFailed(ctx, videoID, store.Locator{Kind: "pending"}, "user", cause); err != nil {
		s.log.Error("recording index failure", "video", videoID, "err", err)
	}
}

// slugFor turns a submitted URL into the readable half of a database key.
//
// It is not the locator -- index.py decides that, because it is the side that
// knows what the link actually resolved to. This only has to produce something
// stable and legible, so that re-adding the same link lands on the same row
// instead of a second copy.
func slugFor(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", fmt.Errorf("no url given")
	}
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return "", fmt.Errorf("url must be http or https")
	}
	if u.Host == "" {
		return "", fmt.Errorf("url has no host")
	}

	if id := youTubeID(u); id != "" {
		return safeSlug(id), nil
	}
	name := strings.TrimSuffix(path.Base(u.Path), filepath.Ext(u.Path))
	if name == "" || name == "/" || name == "." {
		name = fmt.Sprintf("video-%d", time.Now().Unix())
	}
	return safeSlug(name), nil
}

// safeSlug keeps the id printable and URL-safe: it ends up in a path segment,
// and a submitted filename is not to be trusted with that.
func safeSlug(s string) string {
	out := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9',
			r == '-', r == '_':
			return r
		default:
			return '-'
		}
	}, s)
	out = strings.Trim(out, "-")
	if len(out) > 48 {
		out = out[:48]
	}
	if out == "" {
		out = fmt.Sprintf("video-%d", time.Now().Unix())
	}
	return out
}

// shortHash keeps the owner key itself out of any value a user can see.
func shortHash(session string) string {
	sum := sha256.Sum256([]byte(session))
	return hex.EncodeToString(sum[:4])
}

func youTubeID(u *url.URL) string {
	switch {
	case strings.HasSuffix(u.Host, "youtu.be"):
		return strings.TrimPrefix(u.Path, "/")
	case strings.Contains(u.Host, "youtube.com"):
		return u.Query().Get("v")
	}
	return ""
}
