-- +goose Up
-- Query classifications are no longer cached: the model is called on every
-- search, so that a rewritten query is always current and nothing depends on
-- which queries happened to have been seen before. Leaving the table behind
-- would claim a behaviour the system no longer has.
DROP TABLE IF EXISTS query_intent;

-- +goose Down
CREATE TABLE query_intent (
    query      text PRIMARY KEY,
    class      text NOT NULL,
    confidence real,
    model      text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
