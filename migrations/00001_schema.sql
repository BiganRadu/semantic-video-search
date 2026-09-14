-- +goose Up
--
-- The whole schema, in one file.
--
-- This replaces six incremental migrations that were squashed once the only
-- two databases in existence (local and Aiven) were both already at the final
-- state. Everything below is their combined result, not a history: the
-- `pipeline`, `owner` and `expires_at` columns that were once added by later
-- migrations are declared inline, and a short-lived `query_intent` cache that
-- was added and then removed is simply absent.
--
-- Append from here. Editing this file changes what a fresh database gets
-- without changing any database that already ran it, and the two silently
-- diverge; a new numbered file is the only safe way to make a change.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE videos (
    id            text PRIMARY KEY,       -- corpus: a readable name; user: u_<ownerhash>_<slug>
    locator       jsonb NOT NULL,         -- {"kind":"youtube","id":"…","offset":660}
                                          -- never kind:local; see ARCHITECTURE.md 4.1
    source        text NOT NULL,          -- corpus | user
    title         text,
    duration_s    real NOT NULL,
    fps           real,
    state         text NOT NULL DEFAULT 'registered',   -- registered|indexing|ready|failed
    progress      jsonb,
    last_error    text,

    -- Which stages produced this row. Stored so "what needs reindexing?" is a
    -- query and not a guess: a video indexed before a stage existed is still
    -- 'ready' but is not current.
    pipeline      text,

    -- NULL for the shared example corpus. Otherwise a namespaced owner key --
    -- `anon:<session>` or `user:<id>` -- so an anonymous session id can never
    -- collide with an account id, and signing in is a change of prefix.
    owner         text,
    -- Session-owned videos lapse; account-owned ones do not, so this is NULL
    -- for them.
    expires_at    timestamptz,

    indexed_at    timestamptz,
    last_verified timestamptz,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX videos_state_idx    ON videos (state);
CREATE INDEX videos_pipeline_idx ON videos (pipeline);
CREATE INDEX videos_owner_idx    ON videos (owner)      WHERE owner IS NOT NULL;
CREATE INDEX videos_expiry_idx   ON videos (expires_at) WHERE expires_at IS NOT NULL;

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

    -- Weighted so a caption match outranks a transcript match on the same
    -- words. This is the literal-keyword signal; embeddings blur, which is
    -- wrong for proper nouns and numbers.
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

-- Accounts, so a visitor's videos can follow them off one browser. Optional on
-- purpose: the anonymous session still works and still expires. Signing in
-- only changes which key owns the videos.
CREATE TABLE users (
    id            text PRIMARY KEY,
    email         text        NOT NULL,
    -- Algorithm, cost and salt live in the string with the digest, so the
    -- parameters can be raised later without a migration and without
    -- invalidating everyone's password.
    password_hash text        NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- Addresses differing only by case are the same address to every mail server,
-- and letting both register is an account-takeover vector, not a feature.
CREATE UNIQUE INDEX users_email_key ON users (lower(email));

-- Sessions live in the database rather than in a signed cookie so that signing
-- out actually ends the session, everywhere, immediately.
CREATE TABLE auth_sessions (
    -- The SHA-256 of the cookie value, never the value itself: a leaked
    -- database backup should not be a set of working logins.
    token_hash text        PRIMARY KEY,
    user_id    text        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    user_agent text
);
CREATE INDEX auth_sessions_user_idx   ON auth_sessions (user_id);
CREATE INDEX auth_sessions_expiry_idx ON auth_sessions (expires_at);

-- +goose Down
DROP TABLE IF EXISTS auth_sessions, users, transcript_segments,
                     clip_frames, clip_vectors, clips, videos;
DROP EXTENSION IF EXISTS vector;
