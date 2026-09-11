-- +goose Up
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE videos (
    id            text PRIMARY KEY,       -- QVHighlights: <ytid>_<start>_<end>, == annotation "vid"
    locator       jsonb NOT NULL,         -- {"kind":"youtube","id":"…","offset":660}
                                          -- never kind:local; see docs/ARCHITECTURE.md 4.1
    source        text NOT NULL,          -- qvhighlights | pexels | user
    title         text,
    duration_s    real NOT NULL,
    fps           real,
    state         text NOT NULL DEFAULT 'registered',   -- registered|indexing|ready|failed
    progress      jsonb,
    last_error    text,
    indexed_at    timestamptz,
    last_verified timestamptz,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX videos_state_idx ON videos (state);

CREATE TABLE clips (
    id            bigserial PRIMARY KEY,
    video_id      text NOT NULL REFERENCES videos ON DELETE CASCADE,
    idx           int  NOT NULL,
    start_s       real NOT NULL,          -- excerpt-relative, never absolute YouTube time
    end_s         real NOT NULL,

    caption       text,
    people        jsonb,
    objects       jsonb,
    actions       jsonb,
    setting       text,
    tags_text     text,                   -- flattened for FTS: generated cols must be immutable
    caption_model text,
    caption_state text NOT NULL DEFAULT 'pending',

    speech        text,
    lang          text,

    static_score  real,
    novelty       real,

    fts tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(caption,   '')), 'A') ||
        setweight(to_tsvector('english', coalesce(tags_text, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(speech,    '')), 'C')
    ) STORED,

    UNIQUE (video_id, idx)
);
CREATE INDEX clips_fts_idx   ON clips USING gin (fts);
CREATE INDEX clips_video_idx ON clips (video_id, start_s);

CREATE TABLE clip_vectors (
    clip_id     bigint PRIMARY KEY REFERENCES clips ON DELETE CASCADE,
    visual      vector(1152),             -- SigLIP2 so400m, mean of the frame vectors
    caption_vec vector(1024),             -- bge-m3
    speech_vec  vector(1024)              -- bge-m3, NULL when no speech
);

CREATE TABLE clip_frames (
    clip_id   bigint NOT NULL REFERENCES clips ON DELETE CASCADE,
    frame_idx smallint NOT NULL,
    t_s       real NOT NULL,
    embedding halfvec(1152) NOT NULL,     -- needs pgvector >= 0.7
    PRIMARY KEY (clip_id, frame_idx)
);

CREATE TABLE transcript_segments (
    id             bigserial PRIMARY KEY,
    video_id       text NOT NULL REFERENCES videos ON DELETE CASCADE,
    start_s        real NOT NULL,
    end_s          real NOT NULL,
    text           text NOT NULL,
    lang           text,
    avg_logprob    real,
    no_speech_prob real
);
CREATE INDEX transcript_video_idx ON transcript_segments (video_id, start_s);

-- +goose Down
DROP TABLE IF EXISTS transcript_segments, clip_frames, clip_vectors, clips, videos;
DROP EXTENSION IF EXISTS vector;
