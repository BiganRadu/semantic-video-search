-- +goose Up
-- Videos belong either to the shared example corpus (owner IS NULL) or to one
-- visitor's local session. A session needs no account: the server hands out an
-- opaque id in a cookie, and anything added under it is visible only to that
-- browser until it expires.
ALTER TABLE videos ADD COLUMN owner      text;
ALTER TABLE videos ADD COLUMN expires_at timestamptz;

CREATE INDEX videos_owner_idx ON videos (owner) WHERE owner IS NOT NULL;
CREATE INDEX videos_expiry_idx ON videos (expires_at) WHERE expires_at IS NOT NULL;

-- +goose Down
DROP INDEX IF EXISTS videos_expiry_idx;
DROP INDEX IF EXISTS videos_owner_idx;
ALTER TABLE videos DROP COLUMN IF EXISTS expires_at;
ALTER TABLE videos DROP COLUMN IF EXISTS owner;
