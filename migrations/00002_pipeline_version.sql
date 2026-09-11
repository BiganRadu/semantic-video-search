-- +goose Up
-- Records which indexing stages produced a video's rows, so that after a
-- pipeline change the videos needing reindexing are a query rather than a
-- guess. Existing rows predate the speech stage.
ALTER TABLE videos ADD COLUMN pipeline text;
UPDATE videos SET pipeline = 'visual' WHERE state = 'ready';
CREATE INDEX videos_pipeline_idx ON videos (pipeline);

-- +goose Down
DROP INDEX IF EXISTS videos_pipeline_idx;
ALTER TABLE videos DROP COLUMN IF EXISTS pipeline;
