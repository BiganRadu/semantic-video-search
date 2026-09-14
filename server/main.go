// Command server is the whole Go side of the application. Everything it needs
// lives under server/, grouped by what it does rather than by Go convention —
// there is no cmd/ holding a thin main beside an internal/ holding the real
// code, because the two always change together.
//
//	server/          main.go and migrate.go: startup, flags, subcommands
//	server/api/      the HTTP surface: routing, handlers, sessions, accounts
//	server/auth/     password hashing and credential rules
//	server/store/    everything that touches Postgres
//	server/pyproc/   supervising search.py (warm) and index.py (per job)
//
// This file is only wiring: it reads configuration, opens the things that have
// to be open, hands them to api.New and waits. Anything with a decision in it
// belongs in one of the packages above.
//
// It needs a database always, the search worker to answer queries, and index.py
// only where indexing is enabled. On a deployment without a GPU, index.py need
// not even be present.
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"videosearch/server/api"
	"videosearch/server/pyproc"
	"videosearch/server/store"
)

func main() {
	// Schema migrations ship in the same binary rather than a second command:
	// they need the same DSN and the same embedded SQL, and one binary is one
	// thing to build, copy and run.
	if len(os.Args) > 1 && os.Args[1] == "migrate" {
		runMigrations(os.Args[2:])
		return
	}

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(log)

	addr := flag.String("addr", defaultAddr(), "listen address")
	dsn := flag.String("dsn", envOr("DATABASE_URL", ""), "postgres connection string")
	python := flag.String("python", envOr("PYTHON", defaultPython()), "python interpreter")
	pyDir := flag.String("python-dir", envOr("PYTHON_DIR", "python"), "directory holding index.py and search.py")
	indexing := flag.Bool("indexing", envBool("INDEXING_ENABLED", true), "enable the indexing path")
	remote := flag.Bool("remote", envBool("REMOTE_MODELS", false),
		"run the models on Kaggle instead of this machine (for hosts with no RAM for them)")
	token := flag.String("auth-token", os.Getenv("AUTH_TOKEN"), "bearer token for write endpoints")
	static := flag.String("static", envOr("STATIC_DIR", "web/dist"), "built frontend to serve, if present")
	origins := flag.String("cors", envOr("CORS_ORIGINS", "http://localhost:5173"), "comma-separated allowed origins")
	flag.Parse()

	if *dsn == "" {
		log.Error("no DSN (pass -dsn or set DATABASE_URL)")
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	st, err := store.Open(ctx, *dsn)
	if err != nil {
		log.Error("database", "err", err)
		os.Exit(1)
	}
	defer st.Close()

	// Any row still marked 'indexing' belongs to a process that is gone: jobs
	// are held in memory, so nothing survived the restart that claimed them.
	if n, err := st.ReleaseInterruptedIndexes(ctx); err != nil {
		log.Warn("releasing interrupted indexes", "err", err)
	} else if n > 0 {
		log.Info("released interrupted indexes", "videos", n)
	}

	searchScript := filepath.Join(*pyDir, "search.py")
	indexScript := filepath.Join(*pyDir, "index.py")

	// Indexing is disabled outright if index.py is not shipped, regardless of
	// the flag -- a deployment without the script cannot index.
	if *indexing {
		if _, err := os.Stat(indexScript); err != nil {
			log.Warn("index.py not found; indexing disabled", "path", indexScript)
			*indexing = false
		}
	}
	if *remote {
		// Worth saying out loud: in this mode nothing is computed here, and a
		// query costs minutes rather than milliseconds.
		log.Info("remote models: search and indexing run on Kaggle kernels")
	}

	searcher := pyproc.NewSearcher(*python, searchScript, log, *remote)
	log.Info("starting search worker", "script", searchScript)
	if err := searcher.Start(ctx); err != nil {
		log.Error("search worker", "err", err)
		os.Exit(1)
	}
	defer searcher.Close()

	staticDir := *static
	if _, err := os.Stat(filepath.Join(staticDir, "index.html")); err != nil {
		log.Info("no built frontend; serving API only", "looked_in", staticDir)
		staticDir = ""
	}

	srv := api.New(
		api.Config{
			IndexingEnabled: *indexing,
			AuthToken:       *token,
			AllowedOrigins:  strings.Split(*origins, ","),
			StaticDir:       staticDir,
		},
		st, searcher, pyproc.NewIndexer(*python, indexScript, log, *remote), log,
	)

	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           srv.Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(shutdown)
	}()

	log.Info("listening", "addr", "http://"+*addr, "indexing", *indexing)
	if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Error("serve", "err", err)
		os.Exit(1)
	}
}

// defaultPython prefers a venv beside the repo, falling back to the shared one
// under $HOME. The project directory is often on a filesystem where a venv does
// not belong, so the home location is the normal case.
func defaultPython() string {
	candidates := []string{".venv/bin/python"}
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates, filepath.Join(home, ".venvs", "video-search", "bin", "python"))
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return "python3"
}

// defaultAddr binds where the host expects. Platforms that assign a port set
// PORT and require 0.0.0.0; local runs stay on loopback so a dev server is not
// exposed to the network by accident.
func defaultAddr() string {
	if port := os.Getenv("PORT"); port != "" {
		return "0.0.0.0:" + port
	}
	return envOr("ADDR", "127.0.0.1:8080")
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envBool(k string, def bool) bool {
	if v := os.Getenv(k); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}
