package api

import (
	"sync"
	"testing"
	"time"
)

func start(js *jobs, id string, run func(j *job)) (*job, bool) {
	return js.startOrAttach(id, "owner", "", run)
}

// The whole point of detaching: work must not stop when the watcher does.
func TestJobRunsToCompletionAfterEveryoneDetaches(t *testing.T) {
	js := newJobs()
	release := make(chan struct{})
	finished := make(chan struct{})

	j, started := start(js, "v1", func(j *job) {
		j.emit("stage-1")
		<-release // still working while the viewer leaves
		j.emit("stage-2")
		j.emit("done")
		close(finished)
	})
	if !started {
		t.Fatal("first call must start the job")
	}

	_, ch := j.attach()
	j.detach(ch) // the browser goes away mid-index

	close(release)
	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("job stopped when its subscriber left")
	}
	waitDone(t, j)

	if got := j.since(0); len(got) != 3 {
		t.Fatalf("want all 3 events recorded, got %v", got)
	}
}

// One at a time. Two indexes at once means two copies of every model resident
// on one GPU, which fails both instead of finishing either.
func TestOnlyOneJobRunsAtATime(t *testing.T) {
	js := newJobs()
	var mu sync.Mutex
	running, peak := 0, 0
	release := make(chan struct{})

	body := func(j *job) {
		mu.Lock()
		running++
		if running > peak {
			peak = running
		}
		mu.Unlock()

		<-release

		mu.Lock()
		running--
		mu.Unlock()
	}

	var js2 []*job
	for _, id := range []string{"a", "b", "c"} {
		j, _ := start(js, id, body)
		js2 = append(js2, j)
	}

	// Give the worker every chance to start more than one, then let them go.
	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	if peak > 1 {
		mu.Unlock()
		t.Fatalf("%d jobs ran concurrently; indexing must be serialised", peak)
	}
	mu.Unlock()

	close(release)
	for _, j := range js2 {
		waitDone(t, j)
	}

	mu.Lock()
	defer mu.Unlock()
	if peak != 1 {
		t.Fatalf("peak concurrency was %d, want exactly 1", peak)
	}
}

// Waiting jobs must say so, and say where they are in the line.
func TestQueuedJobsReportTheirPosition(t *testing.T) {
	js := newJobs()
	release := make(chan struct{})
	body := func(j *job) { <-release }

	start(js, "a", body)
	start(js, "b", body)
	start(js, "c", body)

	// Let the worker pick up the first one.
	deadline := time.Now().Add(2 * time.Second)
	var views []JobView
	for time.Now().Before(deadline) {
		views = js.viewFor("owner")
		if len(views) == 3 && countState(views, jobIndexing) == 1 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}

	if n := countState(views, jobIndexing); n != 1 {
		t.Fatalf("want exactly 1 running, got %d in %v", n, views)
	}
	if n := countState(views, jobQueued); n != 2 {
		t.Fatalf("want 2 queued, got %d in %v", n, views)
	}
	for _, v := range views {
		if v.State == jobQueued && v.Position < 1 {
			t.Fatalf("a queued job must report its place in the line: %+v", v)
		}
		if v.State == jobIndexing && v.Position != 0 {
			t.Fatalf("a running job has no queue position: %+v", v)
		}
	}
	close(release)
}

// The panel is "what am I waiting on", not "what is the server doing".
func TestJobsAreScopedToTheirOwner(t *testing.T) {
	js := newJobs()
	release := make(chan struct{})
	js.startOrAttach("mine", "alice", "", func(j *job) { <-release })
	js.startOrAttach("theirs", "bob", "", func(j *job) { <-release })

	if got := js.viewFor("alice"); len(got) != 1 || got[0].ID != "mine" {
		t.Fatalf("alice must see only her own job, got %v", got)
	}
	if got := js.viewFor("bob"); len(got) != 1 || got[0].ID != "theirs" {
		t.Fatalf("bob must see only his own job, got %v", got)
	}
	if got := js.viewFor("carol"); len(got) != 0 {
		t.Fatalf("a visitor with no jobs must see none, got %v", got)
	}
	close(release)
}

// A refresh mid-index must watch the running job, not start a second one.
func TestSecondRequestAttachesInsteadOfStartingAgain(t *testing.T) {
	js := newJobs()
	release := make(chan struct{})
	var runs int
	var mu sync.Mutex

	run := func(j *job) {
		mu.Lock()
		runs++
		mu.Unlock()
		<-release
		j.emit("done")
	}

	j1, started1 := start(js, "v1", run)
	j2, started2 := start(js, "v1", run)

	if !started1 || started2 {
		t.Fatalf("want start then attach, got %v %v", started1, started2)
	}
	if j1 != j2 {
		t.Fatal("the second request must get the same job")
	}
	close(release)
	waitDone(t, j1)

	mu.Lock()
	defer mu.Unlock()
	if runs != 1 {
		t.Fatalf("the indexer ran %d times; a refresh must not reindex", runs)
	}
}

// Attaching late must show the stage the job is on, not an empty page.
func TestLateSubscriberReplaysHistory(t *testing.T) {
	js := newJobs()
	release := make(chan struct{})
	j, _ := start(js, "v1", func(j *job) {
		j.emit("download")
		j.emit("frames")
		<-release
		j.emit("done")
	})

	for len(j.since(0)) < 2 {
		time.Sleep(time.Millisecond)
	}
	history, ch := j.attach()
	if len(history) != 2 || history[0] != "download" {
		t.Fatalf("late subscriber lost the earlier stages: %v", history)
	}
	if ch == nil {
		t.Fatal("a running job must still deliver what happens next")
	}

	close(release)
	select {
	case ev := <-ch:
		if ev != "done" {
			t.Fatalf("want the next event, got %v", ev)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("subscriber never received the final event")
	}
}

// A subscriber that stopped reading must still end up with the terminal event,
// which is the difference between "finished" and "hung" on the page.
func TestSinceRecoversEventsDroppedWhileNotReading(t *testing.T) {
	j := newJob("v1", "owner", "")
	_, ch := j.attach()
	for i := 0; i < 300; i++ { // more than the channel buffer holds
		j.emit(i)
	}
	j.emit("done")

	drained := 0
	for {
		select {
		case <-ch:
			drained++
			continue
		default:
		}
		break
	}
	missed := j.since(drained)
	if len(missed) == 0 || missed[len(missed)-1] != "done" {
		t.Fatalf("the terminal event must be recoverable from history, got %v", missed)
	}
}

// Attaching after the end is a normal case: the answer is the whole history and
// no channel, not a subscriber that waits forever.
func TestAttachAfterFinishReturnsHistoryAndNoChannel(t *testing.T) {
	js := newJobs()
	j, _ := start(js, "v1", func(j *job) { j.emit("done") })
	waitDone(t, j)

	history, ch := j.attach()
	if ch != nil {
		t.Fatal("a finished job must not hand out a channel to wait on")
	}
	if len(history) != 1 || history[0] != "done" {
		t.Fatalf("want the finished job's history, got %v", history)
	}
}

// The id has to be released, or a video that failed could never be retried
// without restarting the server.
func TestIdIsFreedSoTheVideoCanBeAddedAgain(t *testing.T) {
	js := newJobs()
	j, _ := start(js, "v1", func(j *job) { j.emit("done") })
	waitDone(t, j)

	j2, started := start(js, "v1", func(j *job) { j.emit("done") })
	if !started {
		t.Fatal("a finished job must not block a retry")
	}
	if j2 == j {
		t.Fatal("a retry must be a new job, not the finished one")
	}
	waitDone(t, j2)
}

// A job that ends without a terminal event died rather than finished, and the
// panel must not show it as still running for ever.
func TestJobEndingWithoutATerminalEventIsReportedFailed(t *testing.T) {
	js := newJobs()
	j, _ := start(js, "v1", func(j *job) {
		j.emit(map[string]any{"event": "progress", "stage": "caption"})
	})
	waitDone(t, j)

	v := j.view(0)
	if v.State != jobFailed || v.Error == "" {
		t.Fatalf("want a failed job with a reason, got %+v", v)
	}
}

// A panicking indexer must not strand every subscriber or wedge the id.
func TestPanicStillReleasesSubscribersAndFreesTheId(t *testing.T) {
	js := newJobs()
	j, _ := start(js, "v1", func(j *job) { panic("boom") })

	_, ch := j.attach()
	if ch != nil {
		select {
		case <-ch: // closed or delivered; either means we were released
		case <-time.After(2 * time.Second):
			t.Fatal("panic left subscribers waiting forever")
		}
	}
	waitDone(t, j)
	if _, started := start(js, "v1", func(j *job) { j.emit("done") }); !started {
		t.Fatal("panic left the id claimed")
	}
}

func countState(views []JobView, state string) int {
	n := 0
	for _, v := range views {
		if v.State == state {
			n++
		}
	}
	return n
}

func waitDone(t *testing.T, j *job) {
	t.Helper()
	for i := 0; i < 3000; i++ {
		if j.isFinished() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("job never finished")
}
