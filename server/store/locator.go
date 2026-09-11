package store

import (
	"encoding/json"
	"fmt"
)

// Kind enumerates where playback points. There is deliberately no "local":
// a locator is persisted forever and must resolve from any deployment, so a
// path that exists only on the indexing machine is never stored.
// See ARCHITECTURE.md 4.1.
type Kind string

const (
	KindYouTube Kind = "youtube"
	KindHTTP    Kind = "http"
)

// Locator is where a video plays back from. Stored in videos.locator.
//
// Offset is the position of the indexed excerpt within the original video, in
// seconds. Every timestamp elsewhere in the system (clips.start_s,
// transcript_segments.start_s) is relative to the excerpt; Offset is added
// exactly once, in the frontend, when building a player URL.
type Locator struct {
	Kind   Kind    `json:"kind"`
	ID     string  `json:"id,omitempty"`  // youtube video id
	URL    string  `json:"url,omitempty"` // http
	Offset float64 `json:"offset,omitempty"`
}

func YouTube(id string, offset float64) Locator {
	return Locator{Kind: KindYouTube, ID: id, Offset: offset}
}

func HTTP(url string) Locator {
	return Locator{Kind: KindHTTP, URL: url}
}

func (l Locator) Validate() error {
	switch l.Kind {
	case KindYouTube:
		if l.ID == "" {
			return fmt.Errorf("youtube locator: empty id")
		}
	case KindHTTP:
		if l.URL == "" {
			return fmt.Errorf("http locator: empty url")
		}
	case "":
		return fmt.Errorf("locator: missing kind")
	default:
		return fmt.Errorf("locator: unsupported kind %q (only youtube and http are persistable)", l.Kind)
	}
	if l.Offset < 0 {
		return fmt.Errorf("locator: negative offset %v", l.Offset)
	}
	return nil
}

func (l Locator) Value() ([]byte, error) { return json.Marshal(l) }
