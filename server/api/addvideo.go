package api

import (
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
)

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
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	send := func(payload any) {
		b, _ := json.Marshal(payload)
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	result, err := s.indexer.Run(r.Context(), req.URL, videoID, func(ev pyproc.Event) {
		send(ev)
	})
	if err != nil {
		send(map[string]any{"event": "error", "message": err.Error()})
		return
	}

	result.Source = "user"
	// index.py reports the video's own title; a title typed on the form wins,
	// because someone who bothered to name it meant it.
	if req.Title != "" {
		result.Title = req.Title
	}
	result.Owner = &owner

	// A video kept by an account is kept until its owner deletes it; one added
	// without an account rides on the anonymous session and lapses with it.
	var expires *time.Time
	if userFrom(r) == nil {
		lapse := time.Now().Add(videoTTL)
		expires = &lapse
	}
	result.ExpiresAt = expires

	send(map[string]any{"event": "progress", "stage": "save"})
	if err := s.store.Save(r.Context(), result); err != nil {
		send(map[string]any{"event": "error", "message": err.Error()})
		return
	}

	done := map[string]any{
		"event": "done", "video_id": videoID,
		"clips": len(result.Clips), "duration_s": result.DurationS,
	}
	if expires != nil {
		done["expires_at"] = expires.Format(time.RFC3339)
	}
	send(done)
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
