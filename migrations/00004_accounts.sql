-- +goose Up
-- Accounts, so a visitor's videos can follow them off one browser.
--
-- An account is optional on purpose: the anonymous session from 00003 still
-- works and still expires. Signing in just changes which key owns the videos,
-- from `anon:<session>` to `user:<id>` -- which is why videos.owner is a
-- namespaced key rather than a bare id, and why an anonymous id can never
-- collide with an account id.
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
DROP TABLE IF EXISTS auth_sessions;
DROP TABLE IF EXISTS users;
