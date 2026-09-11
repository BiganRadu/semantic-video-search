package pyproc

import (
	"bytes"
	"log/slog"
)

// logWriter forwards a subprocess's stderr into the server's logs, line by
// line, so a Python traceback is visible without hunting for a separate file.
type logWriter struct {
	log    *slog.Logger
	source string
	buf    bytes.Buffer
}

func newLogWriter(log *slog.Logger, source string) *logWriter {
	return &logWriter{log: log, source: source}
}

func (w *logWriter) Write(p []byte) (int, error) {
	w.buf.Write(p)
	for {
		line, err := w.buf.ReadBytes('\n')
		if err != nil {
			w.buf.Write(line) // partial line, wait for the rest
			return len(p), nil
		}
		if trimmed := bytes.TrimSpace(line); len(trimmed) > 0 {
			w.log.Info(string(trimmed), "source", w.source)
		}
	}
}
