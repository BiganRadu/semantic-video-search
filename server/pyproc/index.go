package pyproc

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os/exec"

	"videosearch/server/store"
)

// Event is one line of index.py's output.
type Event struct {
	Event   string `json:"event"`
	Stage   string `json:"stage,omitempty"`
	Done    int    `json:"done,omitempty"`
	Total   int    `json:"total,omitempty"`
	Message string `json:"message,omitempty"`
}

// Indexer runs index.py once per video.
type Indexer struct {
	python string
	script string
	log    *slog.Logger
}

func NewIndexer(python, script string, log *slog.Logger) *Indexer {
	return &Indexer{python: python, script: script, log: log}
}

type wireClip struct {
	Idx         int      `json:"idx"`
	StartS      float64  `json:"start_s"`
	EndS        float64  `json:"end_s"`
	Visual      string   `json:"visual"`
	SpeechVec   *string  `json:"speech_vec"`
	CaptionVec  *string  `json:"caption_vec"`
	Speech      *string  `json:"speech"`
	Lang        *string  `json:"lang"`
	Caption     *string  `json:"caption"`
	People      []string `json:"people"`
	Objects     []string `json:"objects"`
	Actions     []string `json:"actions"`
	Setting     *string  `json:"setting"`
	TagsText    *string  `json:"tags_text"`
	StaticScore *float64 `json:"static_score"`
	Novelty     *float64 `json:"novelty"`
	Frames      []struct {
		Idx       int     `json:"idx"`
		TS        float64 `json:"t_s"`
		Embedding string  `json:"embedding"`
	} `json:"frames"`
}

type wireResult struct {
	Video struct {
		ID        string  `json:"id"`
		DurationS float64 `json:"duration_s"`
		FPS       float64 `json:"fps"`
		// Set only when index.py fetched the video itself. These are the two
		// things only the fetch step knows: where it plays back from once the
		// local file is gone, and what it is called.
		Locator *store.Locator `json:"locator"`
		Title   string         `json:"title"`
	} `json:"video"`
	Clips        []wireClip                `json:"clips"`
	Language     *string                   `json:"language"`
	Transcript   []store.TranscriptSegment `json:"transcript"`
	Pipeline     string                    `json:"pipeline"`
	CaptionModel *string                   `json:"caption_model"`
	Stats        struct {
		Clips  int     `json:"clips"`
		Frames int     `json:"frames"`
		TookS  float64 `json:"took_s"`
	} `json:"stats"`
}

// Run indexes the video at a URL: index.py fetches it with yt-dlp, indexes it,
// deletes the bytes, and returns the clips plus the locator to persist.
//
// Go deliberately never handles the video file. Downloading here would mean
// two places that know how to turn a link into a playable stream, and the one
// that does not have yt-dlp would be the one deciding what the locator is.
func (ix *Indexer) Run(ctx context.Context, url, videoID string, onEvent func(Event)) (store.IndexedVideo, error) {
	// index.py also takes --source for a file already on disk, but nothing in
	// Go uses it: the bulk corpus script is Python and imports index_video
	// directly, because loading SigLIP once beats loading it per video.
	cmd := exec.CommandContext(ctx, ix.python, ix.script, "--url", url, "--video-id", videoID)
	cmd.Stderr = newLogWriter(ix.log, "index.py")

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return store.IndexedVideo{}, err
	}
	if err := cmd.Start(); err != nil {
		return store.IndexedVideo{}, fmt.Errorf("start index.py: %w", err)
	}

	var result *wireResult
	var failure string

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 1<<20), 64<<20) // a video's vectors are megabytes
	for scanner.Scan() {
		line := scanner.Bytes()

		var ev Event
		if err := json.Unmarshal(line, &ev); err != nil {
			ix.log.Warn("index.py: unparseable line", "line", string(line))
			continue
		}
		switch ev.Event {
		case "progress":
			if onEvent != nil {
				onEvent(ev)
			}
		case "error":
			failure = ev.Message
		case "result":
			var r wireResult
			if err := json.Unmarshal(line, &r); err != nil {
				return store.IndexedVideo{}, fmt.Errorf("index.py: bad result: %w", err)
			}
			result = &r
		}
	}
	if err := scanner.Err(); err != nil {
		return store.IndexedVideo{}, fmt.Errorf("read index.py: %w", err)
	}
	if err := cmd.Wait(); err != nil && failure == "" {
		return store.IndexedVideo{}, fmt.Errorf("index.py: %w", err)
	}
	if failure != "" {
		return store.IndexedVideo{}, fmt.Errorf("index.py: %s", failure)
	}
	if result == nil {
		return store.IndexedVideo{}, fmt.Errorf("index.py produced no result")
	}

	return toIndexedVideo(result)
}

func toIndexedVideo(r *wireResult) (store.IndexedVideo, error) {
	out := store.IndexedVideo{
		ID:           r.Video.ID,
		DurationS:    r.Video.DurationS,
		FPS:          r.Video.FPS,
		Title:        r.Video.Title,
		Language:     r.Language,
		Pipeline:     r.Pipeline,
		CaptionModel: r.CaptionModel,
		Transcript:   r.Transcript,
		Clips:        make([]store.IndexedClip, 0, len(r.Clips)),
	}
	if r.Video.Locator != nil {
		out.Locator = *r.Video.Locator
	}
	for _, c := range r.Clips {
		clip := store.IndexedClip{
			Idx: c.Idx, StartS: c.StartS, EndS: c.EndS,
			StaticScore: c.StaticScore, Novelty: c.Novelty,
			Speech: c.Speech, Lang: c.Lang,
			Caption: c.Caption, People: c.People, Objects: c.Objects,
			Actions: c.Actions, Setting: c.Setting, TagsText: c.TagsText,
		}
		if c.CaptionVec != nil && *c.CaptionVec != "" {
			v, err := decodeVector(*c.CaptionVec)
			if err != nil {
				return out, fmt.Errorf("clip %d caption: %w", c.Idx, err)
			}
			clip.CaptionVec = v
		}
		if c.SpeechVec != nil && *c.SpeechVec != "" {
			v, err := decodeVector(*c.SpeechVec)
			if err != nil {
				return out, fmt.Errorf("clip %d speech: %w", c.Idx, err)
			}
			clip.SpeechVec = v
		}
		if c.Visual != "" {
			v, err := decodeVector(c.Visual)
			if err != nil {
				return out, fmt.Errorf("clip %d visual: %w", c.Idx, err)
			}
			clip.Visual = v
		}
		for _, f := range c.Frames {
			v, err := decodeVector(f.Embedding)
			if err != nil {
				return out, fmt.Errorf("clip %d frame %d: %w", c.Idx, f.Idx, err)
			}
			clip.Frames = append(clip.Frames, store.IndexedFrame{Idx: f.Idx, T: f.TS, Embedding: v})
		}
		out.Clips = append(out.Clips, clip)
	}
	return out, nil
}

// decodeVector reads base64 little-endian float32, the format index.py emits.
// Vectors travel as base64 rather than JSON numbers because a video carries
// ~170k floats and JSON text costs more to write and parse than the model
// forward pass that produced them.
func decodeVector(s string) ([]float32, error) {
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, err
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("%d bytes is not a whole number of float32", len(raw))
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}
