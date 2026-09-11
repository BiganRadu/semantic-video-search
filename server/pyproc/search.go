// Package pyproc runs the Python scripts that do the actual work.
//
// There are two, and they are started differently on purpose:
//
//	search.py  starts once and stays warm. A cold start costs ~9.5s (torch,
//	           transformers, first forward pass), which would otherwise be
//	           paid on every query.
//	index.py   starts per job. A video takes minutes, so startup is noise.
package pyproc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"sync"
	"time"
)

// Searcher is a long-lived search.py process, spoken to in JSON lines.
type Searcher struct {
	python string
	script string
	log    *slog.Logger

	mu     sync.Mutex // one request at a time; the protocol is a single pipe
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
}

func NewSearcher(python, script string, log *slog.Logger) *Searcher {
	return &Searcher{python: python, script: script, log: log}
}

// Start launches the worker and waits for it to report readiness.
func (s *Searcher) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.startLocked(ctx)
}

func (s *Searcher) startLocked(ctx context.Context) error {
	cmd := exec.Command(s.python, s.script, "--serve")
	cmd.Stderr = newLogWriter(s.log, "search.py")

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start search.py: %w", err)
	}

	s.cmd, s.stdin, s.stdout = cmd, stdin, bufio.NewReaderSize(stdout, 1<<20)

	// The worker announces itself once the model is loaded.
	ready := make(chan error, 1)
	go func() {
		line, err := s.stdout.ReadBytes('\n')
		if err != nil {
			ready <- err
			return
		}
		var hello struct {
			Event string `json:"event"`
		}
		if err := json.Unmarshal(line, &hello); err != nil || hello.Event != "ready" {
			ready <- fmt.Errorf("search.py: unexpected greeting: %s", line)
			return
		}
		ready <- nil
	}()

	select {
	case err := <-ready:
		if err != nil {
			_ = cmd.Process.Kill()
			return fmt.Errorf("search.py failed to start: %w", err)
		}
		s.log.Info("search worker ready")
		return nil
	case <-time.After(3 * time.Minute):
		_ = cmd.Process.Kill()
		return errors.New("search.py did not become ready within 3m")
	case <-ctx.Done():
		_ = cmd.Process.Kill()
		return ctx.Err()
	}
}

// Search sends one request and returns the worker's raw JSON response.
//
// If the worker has died, it is restarted once and the request retried: a
// crashed model process should not take the whole server down.
func (s *Searcher) Search(ctx context.Context, req any) (json.RawMessage, error) {
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	resp, err := s.roundTrip(payload)
	if err == nil {
		return resp, nil
	}

	s.log.Warn("search worker failed, restarting", "err", err)
	if s.cmd != nil && s.cmd.Process != nil {
		_ = s.cmd.Process.Kill()
		_ = s.cmd.Wait()
	}
	if rErr := s.startLocked(ctx); rErr != nil {
		return nil, fmt.Errorf("%w (restart also failed: %v)", err, rErr)
	}
	return s.roundTrip(payload)
}

func (s *Searcher) roundTrip(payload []byte) (json.RawMessage, error) {
	if s.stdin == nil || s.stdout == nil {
		return nil, errors.New("search worker not running")
	}
	if _, err := s.stdin.Write(append(payload, '\n')); err != nil {
		return nil, fmt.Errorf("write to search.py: %w", err)
	}
	line, err := s.stdout.ReadBytes('\n')
	if err != nil {
		return nil, fmt.Errorf("read from search.py: %w", err)
	}
	return json.RawMessage(line), nil
}

func (s *Searcher) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stdin != nil {
		_ = s.stdin.Close()
	}
	if s.cmd != nil && s.cmd.Process != nil {
		_ = s.cmd.Process.Kill()
		return s.cmd.Wait()
	}
	return nil
}
