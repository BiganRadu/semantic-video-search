-- +goose Up
-- Cached query classifications.
--
-- A remote model decides whether a query is answered by what is seen or by what
-- is said, and the answer sets the fusion weights. The call is far too slow to
-- sit on the search path -- measured against NVIDIA NIM, p50 was over twenty
-- seconds with a fast path around 260ms -- so a search never waits for it.
-- Instead the first search of a query answers with default weights and warms
-- this table in the background; every later search of the same query, by anyone,
-- is routed.
--
-- That makes the cache the feature, not an optimisation: without it the
-- classifier would be unusable.
CREATE TABLE query_intent (
    query      text PRIMARY KEY,   -- normalized: lowercased, whitespace collapsed
    class      text NOT NULL,      -- visual | speech | mixed
    confidence real,
    model      text NOT NULL,      -- which model said so, so a change can be re-run
    created_at timestamptz NOT NULL DEFAULT now()
);

-- +goose Down
DROP TABLE IF EXISTS query_intent;
