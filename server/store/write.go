package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// IndexedFrame is one sampled still and its visual embedding.
type IndexedFrame struct {
	Idx       int
	T         float64
	Embedding []float32
}

// IndexedClip is a clip ready to be persisted. Timestamps are relative to the
// excerpt, never absolute source time.
type IndexedClip struct {
	Idx         int
	StartS      float64
	EndS        float64
	StaticScore *float64
	Novelty     *float64
	Visual      []float32
	SpeechVec   []float32
	CaptionVec  []float32
	Speech      *string
	Lang        *string
	Caption     *string
	People      []string
	Objects     []string
	Actions     []string
	Setting     *string
	TagsText    *string
	Frames      []IndexedFrame
}

// TranscriptSegment keeps Whisper's own boundaries, not the clip grid: they are
// the raw signal, and the player wants them intact.
type TranscriptSegment struct {
	StartS       float64 `json:"start_s"`
	EndS         float64 `json:"end_s"`
	Text         string  `json:"text"`
	AvgLogprob   float64 `json:"avg_logprob"`
	NoSpeechProb float64 `json:"no_speech_prob"`
}

// IndexedVideo is the complete result of indexing one video. It is written in a
// single transaction: the video is the unit of atomicity, so a half-indexed
// video is never visible.
type IndexedVideo struct {
	ID           string
	Locator      Locator
	Source       string
	Title        string
	DurationS    float64
	FPS          float64
	Language     *string
	Pipeline     string
	CaptionModel *string
	Owner        *string    // NULL for the shared example corpus
	ExpiresAt    *time.Time // NULL means it never lapses
	Clips        []IndexedClip
	Transcript   []TranscriptSegment
}

// Save writes a fully indexed video, replacing any previous index of it.
func (s *Store) Save(ctx context.Context, v IndexedVideo) error {
	if err := v.Locator.Validate(); err != nil {
		return fmt.Errorf("save %s: %w", v.ID, err)
	}
	loc, err := v.Locator.Value()
	if err != nil {
		return err
	}

	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(context.WithoutCancel(ctx))

	if _, err := tx.Exec(ctx, `
		INSERT INTO videos (id, locator, source, title, duration_s, fps, state,
		                    indexed_at, pipeline, owner, expires_at, progress, last_error)
		VALUES ($1, $2, $3, NULLIF($4,''), $5, $6, 'ready', now(), $7, $8, $9, NULL, NULL)
		ON CONFLICT (id) DO UPDATE SET
			locator = EXCLUDED.locator, source = EXCLUDED.source, title = EXCLUDED.title,
			duration_s = EXCLUDED.duration_s, fps = EXCLUDED.fps,
			state = 'ready', indexed_at = now(), pipeline = EXCLUDED.pipeline,
			owner = EXCLUDED.owner, expires_at = EXCLUDED.expires_at,
			progress = NULL, last_error = NULL`,
		v.ID, loc, v.Source, v.Title, v.DurationS, v.FPS, v.Pipeline,
		v.Owner, v.ExpiresAt); err != nil {
		return fmt.Errorf("save %s: video: %w", v.ID, err)
	}

	// Reindexing replaces everything derived from the video. Clip vectors and
	// frames cascade from clips; transcripts hang off the video directly.
	if _, err := tx.Exec(ctx, `DELETE FROM clips WHERE video_id = $1`, v.ID); err != nil {
		return fmt.Errorf("save %s: clear clips: %w", v.ID, err)
	}
	if _, err := tx.Exec(ctx, `DELETE FROM transcript_segments WHERE video_id = $1`, v.ID); err != nil {
		return fmt.Errorf("save %s: clear transcript: %w", v.ID, err)
	}

	if len(v.Transcript) > 0 {
		tb := &pgx.Batch{}
		for _, t := range v.Transcript {
			tb.Queue(`
				INSERT INTO transcript_segments
					(video_id, start_s, end_s, text, lang, avg_logprob, no_speech_prob)
				VALUES ($1,$2,$3,$4,$5,$6,$7)`,
				v.ID, t.StartS, t.EndS, t.Text, v.Language, t.AvgLogprob, t.NoSpeechProb)
		}
		if err := tx.SendBatch(ctx, tb).Close(); err != nil {
			return fmt.Errorf("save %s: transcript: %w", v.ID, err)
		}
	}

	clipIDs := make([]int64, len(v.Clips))
	batch := &pgx.Batch{}
	for _, c := range v.Clips {
		batch.Queue(`
			INSERT INTO clips (video_id, idx, start_s, end_s, static_score, novelty,
			                   speech, lang, caption, people, objects, actions, setting,
			                   tags_text, caption_model, caption_state)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16) RETURNING id`,
			v.ID, c.Idx, c.StartS, c.EndS, c.StaticScore, c.Novelty, c.Speech, c.Lang,
			c.Caption, jsonArray(c.People), jsonArray(c.Objects), jsonArray(c.Actions),
			c.Setting, c.TagsText, v.CaptionModel, captionState(c.Caption))
	}
	br := tx.SendBatch(ctx, batch)
	for i := range v.Clips {
		if err := br.QueryRow().Scan(&clipIDs[i]); err != nil {
			br.Close()
			return fmt.Errorf("save %s: clip %d: %w", v.ID, i, err)
		}
	}
	if err := br.Close(); err != nil {
		return fmt.Errorf("save %s: clips: %w", v.ID, err)
	}

	vecs := &pgx.Batch{}
	for i, c := range v.Clips {
		if c.Visual != nil || c.SpeechVec != nil || c.CaptionVec != nil {
			vecs.Queue(`INSERT INTO clip_vectors (clip_id, visual, speech_vec, caption_vec)
			            VALUES ($1, $2::vector, $3::vector, $4::vector)`,
				clipIDs[i], vecLiteral(c.Visual), vecLiteral(c.SpeechVec), vecLiteral(c.CaptionVec))
		}
		for _, f := range c.Frames {
			vecs.Queue(`INSERT INTO clip_frames (clip_id, frame_idx, t_s, embedding) VALUES ($1,$2,$3,$4::halfvec)`,
				clipIDs[i], f.Idx, f.T, Vec(f.Embedding))
		}
	}
	if vecs.Len() > 0 {
		if err := tx.SendBatch(ctx, vecs).Close(); err != nil {
			return fmt.Errorf("save %s: vectors: %w", v.ID, err)
		}
	}

	return tx.Commit(ctx)
}

// IndexedIDs returns the ids already indexed, so a batch run can resume.
func (s *Store) IndexedIDs(ctx context.Context) (map[string]bool, error) {
	rows, err := s.Pool.Query(ctx, `SELECT id FROM videos WHERE state = 'ready'`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := map[string]bool{}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out[id] = true
	}
	return out, rows.Err()
}

// ReleaseInterruptedIndexes clears rows left mid-index by a process that is no
// longer running. Jobs live in memory, so anything still marked indexing at
// startup was interrupted by definition -- otherwise the row says 'indexing'
// forever and the video can never be retried.
func (s *Store) ReleaseInterruptedIndexes(ctx context.Context) (int64, error) {
	tag, err := s.Pool.Exec(ctx, `
		UPDATE videos SET state = 'failed',
		       last_error = 'indexing was interrupted by a server restart'
		WHERE state = 'indexing'`)
	if err != nil {
		return 0, err
	}
	return tag.RowsAffected(), nil
}

// MarkIndexing claims the row before the work starts, so a job in flight is
// visible and one that died is visible as stuck rather than absent.
//
// Existing columns survive a conflict: re-adding an indexed video must not
// blank its title until the new run has something to replace it with.
func (s *Store) MarkIndexing(ctx context.Context, id string, source, owner string,
	expires *time.Time) error {
	_, err := s.Pool.Exec(ctx, `
		INSERT INTO videos (id, locator, source, duration_s, state, owner, expires_at)
		VALUES ($1, '{"kind":"pending"}'::jsonb, $2, 0, 'indexing', $3, $4)
		ON CONFLICT (id) DO UPDATE SET
			state = 'indexing', last_error = NULL, progress = NULL,
			owner = EXCLUDED.owner, expires_at = EXCLUDED.expires_at`,
		id, source, owner, expires)
	return err
}

// MarkFailed records why a video could not be indexed, so a rerun retries only
// the failures instead of the whole corpus.
func (s *Store) MarkFailed(ctx context.Context, id string, loc Locator, source string, cause error) error {
	raw, err := loc.Value()
	if err != nil {
		return err
	}
	_, err = s.Pool.Exec(ctx, `
		INSERT INTO videos (id, locator, source, duration_s, state, last_error)
		VALUES ($1,$2,$3,0,'failed',$4)
		ON CONFLICT (id) DO UPDATE SET state='failed', last_error=EXCLUDED.last_error`,
		id, raw, source, cause.Error())
	return err
}

// VideoSummary is what the library listing shows.
type VideoSummary struct {
	ID        string  `json:"id"`
	Locator   Locator `json:"locator"`
	Title     string  `json:"title,omitempty"`
	Source    string  `json:"source"`
	DurationS float64 `json:"duration_s"`
	State     string  `json:"state"`
	Clips     int     `json:"clips"`
}

// clipCount is the embedded summary's count, kept distinct from the detail
// type's clip list.

// ListVideos returns one corpus: the shared examples when owner is empty, or a
// single session's videos otherwise. The two never mix.
func (s *Store) ListVideos(ctx context.Context, owner string) ([]VideoSummary, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT v.id, v.locator, coalesce(v.title,''), v.source, v.duration_s, v.state,
		       (SELECT count(*) FROM clips c WHERE c.video_id = v.id)
		FROM videos v
		WHERE (v.expires_at IS NULL OR v.expires_at > now())
		  AND CASE WHEN $1 = '' THEN v.owner IS NULL ELSE v.owner = $1 END
		  -- A video still being indexed is not in the library yet: it has no
		  -- duration, no clips and a placeholder locator, so it would render as
		  -- a broken card. The jobs dock is where work in flight is reported.
		  -- A failed one stays listed, because that is a result worth seeing.
		  AND v.state <> 'indexing'
		ORDER BY v.indexed_at DESC NULLS LAST, v.id`, owner)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []VideoSummary{}
	for rows.Next() {
		var v VideoSummary
		var loc []byte
		if err := rows.Scan(&v.ID, &loc, &v.Title, &v.Source, &v.DurationS, &v.State, &v.Clips); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(loc, &v.Locator); err != nil {
			return nil, fmt.Errorf("video %s: bad locator: %w", v.ID, err)
		}
		out = append(out, v)
	}
	return out, rows.Err()
}

// ErrNotYours means the video does not exist, or exists and belongs to someone
// else. The caller must not distinguish the two: answering "forbidden" would
// confirm that another owner's video exists.
var ErrNotYours = errors.New("no such video")

// DeleteOwnedVideo removes a video only if it belongs to this owner. The
// example corpus has no owner and is therefore never deletable through the API.
func (s *Store) DeleteOwnedVideo(ctx context.Context, id, owner string) error {
	if owner == "" {
		return ErrNotYours
	}
	tag, err := s.Pool.Exec(ctx, `DELETE FROM videos WHERE id = $1 AND owner = $2`, id, owner)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotYours
	}
	return nil
}

// SweepExpired removes session videos past their lifetime.
func (s *Store) SweepExpired(ctx context.Context) (int64, error) {
	tag, err := s.Pool.Exec(ctx,
		`DELETE FROM videos WHERE expires_at IS NOT NULL AND expires_at <= now()`)
	if err != nil {
		return 0, err
	}
	return tag.RowsAffected(), nil
}

// vecLiteral renders a vector for pgvector, or NULL when the signal is absent.
func vecLiteral(v []float32) *string {
	if v == nil {
		return nil
	}
	lit := Vec(v)
	return &lit
}

// jsonArray stores a tag list as jsonb, or NULL when empty. An empty array and
// "no caption ran" are different states and should not look alike.
func jsonArray(items []string) []byte {
	if len(items) == 0 {
		return nil
	}
	raw, err := json.Marshal(items)
	if err != nil {
		return nil
	}
	return raw
}

func captionState(caption *string) string {
	if caption == nil || *caption == "" {
		return "empty"
	}
	return "ok"
}

// ClipDetail is one clip as the player shows it.
type ClipDetail struct {
	ID      int64    `json:"id"`
	Idx     int      `json:"idx"`
	StartS  float64  `json:"start_s"`
	EndS    float64  `json:"end_s"`
	Caption *string  `json:"caption"`
	Speech  *string  `json:"speech"`
	Objects []string `json:"objects,omitempty"`
	Actions []string `json:"actions,omitempty"`
	Setting *string  `json:"setting,omitempty"`
}

// VideoDetail is everything the player needs for one video: the locator to play
// from, the clip timeline, and the transcript. Timestamps stay relative to the
// indexed excerpt; the frontend adds locator.offset exactly once.
type VideoDetail struct {
	VideoSummary
	Language *string `json:"language,omitempty"`
	Pipeline *string `json:"pipeline,omitempty"`
	// Which corpus this video is in, so the player can scope a within-video
	// search correctly. It is derived from ownership, which is the only thing
	// that decides it -- `source` says where a video came from, not whose it
	// is, and guessing from it broke search on any video whose ownership
	// changed after indexing.
	Collection string              `json:"collection"`
	Clips      []ClipDetail        `json:"clips"`
	Transcript []TranscriptSegment `json:"transcript"`
}

// VideoDetail returns a video only if the caller may see it: the shared example
// corpus, or their own session's. Another visitor's upload is not found rather
// than forbidden, so the API does not confirm that it exists.
func (s *Store) VideoDetail(ctx context.Context, id, owner string) (*VideoDetail, error) {
	var v VideoDetail
	var loc []byte
	var shared bool
	err := s.Pool.QueryRow(ctx, `
		SELECT v.id, v.locator, coalesce(v.title,''), v.source, v.duration_s, v.state,
		       v.pipeline, v.owner IS NULL,
		       (SELECT count(*) FROM clips c WHERE c.video_id = v.id)
		FROM videos v
		WHERE v.id = $1
		  AND (v.owner IS NULL OR v.owner = $2)
		  AND (v.expires_at IS NULL OR v.expires_at > now())`, id, owner).
		Scan(&v.ID, &loc, &v.Title, &v.Source, &v.DurationS, &v.State, &v.Pipeline,
			&shared, &v.VideoSummary.Clips)
	if err != nil {
		return nil, err
	}
	v.Collection = "mine"
	if shared {
		v.Collection = "examples"
	}
	if err := json.Unmarshal(loc, &v.Locator); err != nil {
		return nil, fmt.Errorf("video %s: bad locator: %w", id, err)
	}

	rows, err := s.Pool.Query(ctx, `
		SELECT id, idx, start_s, end_s, caption, speech, objects, actions, setting
		FROM clips WHERE video_id = $1 ORDER BY idx`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	v.Clips = []ClipDetail{}
	for rows.Next() {
		var c ClipDetail
		var objects, actions []byte
		if err := rows.Scan(&c.ID, &c.Idx, &c.StartS, &c.EndS, &c.Caption, &c.Speech,
			&objects, &actions, &c.Setting); err != nil {
			return nil, err
		}
		_ = json.Unmarshal(objects, &c.Objects)
		_ = json.Unmarshal(actions, &c.Actions)
		v.Clips = append(v.Clips, c)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	tRows, err := s.Pool.Query(ctx, `
		SELECT start_s, end_s, text, coalesce(avg_logprob,0), coalesce(no_speech_prob,0), lang
		FROM transcript_segments WHERE video_id = $1 ORDER BY start_s`, id)
	if err != nil {
		return nil, err
	}
	defer tRows.Close()

	v.Transcript = []TranscriptSegment{}
	for tRows.Next() {
		var t TranscriptSegment
		if err := tRows.Scan(&t.StartS, &t.EndS, &t.Text, &t.AvgLogprob, &t.NoSpeechProb,
			&v.Language); err != nil {
			return nil, err
		}
		v.Transcript = append(v.Transcript, t)
	}
	return &v, tRows.Err()
}
