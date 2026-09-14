package api

import (
	"sync"
	"time"

	"videosearch/server/pyproc"
)

// Index jobs: detached from the request that started them, and run one at a
// time.
//
// Detached, because indexing used to run on the HTTP request's context, so
// closing the tab cancelled it: the subprocess was killed mid-caption, no row
// had been written yet, and the job vanished without an error anywhere. The
// connection is a viewer of a job now, not its owner.
//
// One at a time, because the models do not share the GPU gracefully. Two
// indexes at once means two copies of Qwen3-VL, SigLIP and Whisper resident
// together, which is how a 24 GB card runs out of memory and fails both jobs
// instead of finishing either. Serialising is also honest: the second video
// was never going to be quicker for having started earlier.

const (
	// How long a finished job stays visible after it ends. Long enough to see
	// "done" on a page you come back to, short enough that the list is what is
	// happening rather than a history.
	jobRetain = 30 * time.Minute
)

// Job states, as reported to the UI.
const (
	jobQueued   = "queued"
	jobIndexing = "indexing"
	jobDone     = "done"
	jobFailed   = "failed"
)

// JobView is one job as the owner's UI sees it.
type JobView struct {
	ID    string `json:"id"`
	Title string `json:"title,omitempty"`
	State string `json:"state"`
	Stage string `json:"stage,omitempty"`
	Done  int    `json:"done,omitempty"`
	Total int    `json:"total,omitempty"`
	// 1-based place in the queue, set only while waiting.
	Position int    `json:"position,omitempty"`
	Error    string `json:"error,omitempty"`
	Started  int64  `json:"started"`
}

// job is one index, plus everything emitted so far.
type job struct {
	id      string
	owner   string
	title   string
	created time.Time
	run     func(j *job) // the indexing work, called by the queue worker

	mu       sync.Mutex
	state    string
	stage    string
	done     int
	total    int
	errMsg   string
	history  []any // replayed to anyone who attaches late
	subs     map[chan any]struct{}
	finished bool
	ended    time.Time
}

func newJob(id, owner, title string) *job {
	return &job{
		id: id, owner: owner, title: title, created: time.Now(),
		state: jobQueued, subs: map[chan any]struct{}{},
	}
}

// emit records an event, updates the reportable status, and offers it to every
// live subscriber.
func (j *job) emit(ev any) {
	j.mu.Lock()
	defer j.mu.Unlock()

	// The status panel and the SSE stream read the same events; keeping one
	// source avoids the two disagreeing about what stage a job is on.
	switch e := ev.(type) {
	case pyproc.Event:
		switch e.Event {
		case "error":
			j.state, j.errMsg = jobFailed, e.Message
		default:
			if e.Stage != "" {
				j.state, j.stage, j.done, j.total = jobIndexing, e.Stage, e.Done, e.Total
			}
		}
	case map[string]any:
		switch e["event"] {
		case "done":
			j.state, j.stage = jobDone, ""
		case "error":
			j.state, _ = jobFailed, ""
			if m, ok := e["message"].(string); ok {
				j.errMsg = m
			}
		}
	}

	j.history = append(j.history, ev)
	for ch := range j.subs {
		// Never block the indexer on a slow reader. A dropped progress tick
		// costs nothing: the subscriber replays from history when it stops,
		// so the record it ends up with is still complete.
		select {
		case ch <- ev:
		default:
		}
	}
}

// setTitle records what is being indexed, once the pipeline has resolved it.
func (j *job) setTitle(title string) {
	if title == "" {
		return
	}
	j.mu.Lock()
	j.title = title
	j.mu.Unlock()
}

// finish closes the job to new events and releases every subscriber.
func (j *job) finish() {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.finished, j.ended = true, time.Now()
	if j.state != jobDone && j.state != jobFailed {
		// Ended without a terminal event: the process died rather than failed.
		j.state = jobFailed
		if j.errMsg == "" {
			j.errMsg = "indexing stopped unexpectedly"
		}
	}
	for ch := range j.subs {
		close(ch)
	}
	j.subs = nil
}

// markRunning is set when the queue picks the job up, not when its first stage
// arrives. Model loading takes a minute before anything is emitted, and a job
// that is actually running must not still read as "queued" during it.
func (j *job) markRunning() {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.state == jobQueued {
		j.state = jobIndexing
	}
}

func (j *job) isFinished() bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.finished
}

// attach returns what has happened already, and a channel for what happens
// next. The channel is nil when the job has already finished.
func (j *job) attach() (history []any, ch chan any) {
	j.mu.Lock()
	defer j.mu.Unlock()
	history = append(history, j.history...)
	if j.finished {
		return history, nil
	}
	ch = make(chan any, 256)
	j.subs[ch] = struct{}{}
	return history, ch
}

func (j *job) detach(ch chan any) {
	j.mu.Lock()
	defer j.mu.Unlock()
	delete(j.subs, ch)
}

// since returns events recorded after the first n, for a subscriber that has
// stopped reading and wants whatever it missed.
func (j *job) since(n int) []any {
	j.mu.Lock()
	defer j.mu.Unlock()
	if n >= len(j.history) {
		return nil
	}
	return append([]any(nil), j.history[n:]...)
}

func (j *job) view(position int) JobView {
	j.mu.Lock()
	defer j.mu.Unlock()
	v := JobView{
		ID: j.id, Title: j.title, State: j.state, Stage: j.stage,
		Done: j.done, Total: j.total, Error: j.errMsg,
		Started: j.created.Unix(),
	}
	if j.state == jobQueued {
		v.Position = position
	}
	return v
}

// jobs is every index this process knows about: one running, the rest waiting,
// plus those that finished recently enough to still be worth showing.
type jobs struct {
	mu      sync.Mutex
	m       map[string]*job
	queue   []*job
	running *job
	wake    chan struct{}
}

func newJobs() *jobs {
	js := &jobs{m: map[string]*job{}, wake: make(chan struct{}, 1)}
	go js.worker()
	return js
}

// startOrAttach returns the job already handling this video, or enqueues one.
//
// The bool reports whether this call created it, which is the difference
// between "queued for indexing" and "you are watching one already under way".
func (js *jobs) startOrAttach(id, owner, title string, run func(j *job)) (*job, bool) {
	js.mu.Lock()
	if existing, ok := js.m[id]; ok && !existing.isFinished() {
		js.mu.Unlock()
		return existing, false
	}
	j := newJob(id, owner, title)
	j.run = run
	js.m[id] = j // a finished job for this id is replaced, so a retry can run
	js.queue = append(js.queue, j)
	js.prune()
	js.mu.Unlock()

	js.nudge()
	return j, true
}

func (js *jobs) nudge() {
	select {
	case js.wake <- struct{}{}:
	default: // already pending; the worker will find the queue either way
	}
}

// worker runs the queue, strictly one job at a time.
func (js *jobs) worker() {
	for range js.wake {
		for {
			js.mu.Lock()
			if js.running != nil || len(js.queue) == 0 {
				js.mu.Unlock()
				break
			}
			j := js.queue[0]
			js.queue = js.queue[1:]
			js.running = j
			js.mu.Unlock()

			j.markRunning()
			js.execute(j)

			js.mu.Lock()
			js.running = nil
			js.mu.Unlock()
		}
	}
}

func (js *jobs) execute(j *job) {
	defer func() {
		// A panic in the indexer must still release the subscribers, or the
		// page waits on a job that will never report anything again.
		if p := recover(); p != nil {
			j.emit(map[string]any{"event": "error", "message": "indexing panicked"})
		}
		j.finish()
	}()
	j.run(j)
}

// prune drops finished jobs that nobody needs to see any more. Called with the
// lock held.
func (js *jobs) prune() {
	cutoff := time.Now().Add(-jobRetain)
	for id, j := range js.m {
		j.mu.Lock()
		stale := j.finished && j.ended.Before(cutoff)
		j.mu.Unlock()
		if stale {
			delete(js.m, id)
		}
	}
}

// viewFor is one owner's jobs, newest first. Jobs are per-owner because the
// panel is "what am I waiting on", not "what is the server doing".
func (js *jobs) viewFor(owner string) []JobView {
	js.mu.Lock()
	defer js.mu.Unlock()
	js.prune()

	place := map[*job]int{}
	for i, j := range js.queue {
		place[j] = i + 1
	}

	out := []JobView{}
	for _, j := range js.m {
		if j.owner != owner {
			continue
		}
		out = append(out, j.view(place[j]))
	}
	// Newest first: the thing just submitted is the thing being watched.
	for i := 0; i < len(out); i++ {
		for k := i + 1; k < len(out); k++ {
			if out[k].Started > out[i].Started {
				out[i], out[k] = out[k], out[i]
			}
		}
	}
	return out
}
