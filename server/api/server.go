// Package api is the HTTP surface: routing, handlers, sessions and accounts.
// It serves no HTML of its own -- the frontend is a React app that calls these
// endpoints, served either by a dev server in development or as static files
// from web/dist in production.
//
//	server.go     router, search and video endpoints
//	session.go    the anonymous session cookie
//	auth.go       accounts: register/login/logout, ownership, rate limiting
//	addvideo.go   upload and indexing progress over SSE
package api

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"

	"videosearch/server/pyproc"
	"videosearch/server/store"
)

type Config struct {
	IndexingEnabled bool
	AuthToken       string   // when set, write endpoints require it
	AllowedOrigins  []string // for the frontend dev server
	StaticDir       string   // optional built frontend
}

type Server struct {
	cfg      Config
	store    *store.Store
	searcher *pyproc.Searcher
	indexer  *pyproc.Indexer
	log      *slog.Logger

	// Sign-in attempts per client address per minute. Guessing a password
	// should cost something even before it reaches the KDF.
	logins *limiter

	// Indexes in flight, so they survive the request that started them.
	jobs *jobs
}

func New(cfg Config, st *store.Store, s *pyproc.Searcher, ix *pyproc.Indexer, log *slog.Logger) *Server {
	return &Server{cfg: cfg, store: st, searcher: s, indexer: ix, log: log,
		logins: newLimiter(10), jobs: newJobs()}
}

func (s *Server) Routes() http.Handler {
	r := chi.NewRouter()
	// withSession runs before withAccount: a signed-in visitor still carries
	// the anonymous id, which is what lets their earlier uploads be claimed.
	r.Use(middleware.RequestID, middleware.RealIP, middleware.Recoverer, s.cors,
		s.withSession, s.withAccount)

	r.Get("/healthz", s.healthz)

	r.Route("/api", func(r chi.Router) {
		r.Get("/config", s.apiConfig)

		// Accounts are optional: everything below works signed out too. An
		// account only makes "your videos" follow you off this browser.
		r.Route("/auth", func(r chi.Router) {
			r.Get("/me", s.apiMe)
			r.Post("/register", s.apiRegister)
			r.Post("/login", s.apiLogin)
			r.Post("/logout", s.apiLogout)
		})

		// Reads are public: the hosted demo is read-only.
		r.Get("/search", s.apiSearch)
		r.Get("/videos", s.apiListVideos)
		r.Get("/jobs", s.apiJobs)
		r.Get("/videos/{id}", s.apiVideoDetail)

		// Writes exist only where indexing does, and can require a token.
		// They always act on the caller's own session, never the example corpus.
		r.Group(func(r chi.Router) {
			r.Use(s.requireIndexing, s.requireAuth)
			r.Post("/videos", s.apiAddVideo)
			r.Delete("/videos/{id}", s.apiDeleteVideo)
		})
	})

	// A built frontend, when one is present. Unknown paths fall back to
	// index.html so client-side routing works on a hard refresh.
	if s.cfg.StaticDir != "" {
		r.NotFound(s.serveStatic)
	}
	return r
}

// --- middleware ------------------------------------------------------------

func (s *Server) cors(next http.Handler) http.Handler {
	allowed := map[string]bool{}
	for _, o := range s.cfg.AllowedOrigins {
		allowed[o] = true
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if origin := r.Header.Get("Origin"); origin != "" && allowed[origin] {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) requireIndexing(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !s.cfg.IndexingEnabled {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{
				"error": "indexing is disabled on this deployment (it needs a GPU)",
			})
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) requireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.AuthToken == "" { // local development: no token configured
			next.ServeHTTP(w, r)
			return
		}
		if strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ") != s.cfg.AuthToken {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// --- endpoints -------------------------------------------------------------

// apiConfig tells the frontend what this deployment can do, so it can disable
// the add-video route rather than hiding it.
func (s *Server) apiConfig(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"indexing_enabled": s.cfg.IndexingEnabled,
		"auth_required":    s.cfg.AuthToken != "",
		"accounts_enabled": true,
		"user":             userFrom(r),
	})
}

func (s *Server) healthz(w http.ResponseWriter, r *http.Request) {
	status := map[string]any{"ok": true, "indexing": s.cfg.IndexingEnabled}
	if err := s.store.Pool.Ping(r.Context()); err != nil {
		status["ok"], status["database"] = false, err.Error()
		writeJSON(w, http.StatusServiceUnavailable, status)
		return
	}
	writeJSON(w, http.StatusOK, status)
}

func (s *Server) apiSearch(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()

	req := map[string]any{
		"q":          q.Get("q"),
		"scope":      firstNonEmpty(q.Get("scope"), "corpus"),
		"collection": firstNonEmpty(q.Get("collection"), "examples"),
		// The namespaced owner key, not the bare session id: videos.owner holds
		// "anon:<session>" or "user:<id>", and search compares against it
		// directly. Sending the raw session here made collection=mine match
		// nothing at all.
		"owner": ownerKey(r),
	}
	if v := q.Get("video_id"); v != "" {
		req["video_id"] = v
	}
	if v := q.Get("signals"); v != "" {
		req["signals"] = v
	}
	if v := q.Get("weights"); v != "" {
		req["weights"] = v
	}
	// Assembly parameters are forwarded verbatim, as strings; search.py owns
	// their defaults and their parsing, so there is one place they are defined.
	for _, name := range []string{"assemble", "moment_gap", "moment_max_len", "moment_decay", "moment_floor", "moment_per_video", "min_relevance", "route"} {
		if v := q.Get(name); v != "" {
			req[name] = v
		}
	}
	if v := q.Get("k"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "k must be an integer"})
			return
		}
		req["k"] = n
	}

	raw, err := s.searcher.Search(r.Context(), req)
	if err != nil {
		s.log.Error("search", "err", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}

	// search.py reports its own failures in-band; pass them through with a
	// status code rather than unwrapping and re-wrapping.
	var probe struct {
		OK bool `json:"ok"`
	}
	_ = json.Unmarshal(raw, &probe)

	w.Header().Set("Content-Type", "application/json")
	if !probe.OK {
		w.WriteHeader(http.StatusBadRequest)
	}
	_, _ = w.Write(raw)
}

// apiJobs is what this visitor is waiting on: their own index jobs, queued,
// running and recently finished.
//
// Scoped to the owner, not the server. Whose videos are being indexed is not
// something one visitor should learn from another's queue -- only the fact
// that something is ahead of them, which the position carries.
func (s *Server) apiJobs(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"jobs": s.jobs.viewFor(ownerKey(r))})
}

func (s *Server) apiListVideos(w http.ResponseWriter, r *http.Request) {
	collection := firstNonEmpty(r.URL.Query().Get("collection"), "examples")
	owner := ""
	if collection == "mine" {
		owner = ownerKey(r)
	}
	videos, err := s.store.ListVideos(r.Context(), owner)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"videos": videos, "collection": collection})
}

func (s *Server) apiVideoDetail(w http.ResponseWriter, r *http.Request) {
	detail, err := s.store.VideoDetail(r.Context(), chi.URLParam(r, "id"), ownerKey(r))
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "no such video"})
		return
	}
	writeJSON(w, http.StatusOK, detail)
}

func (s *Server) apiDeleteVideo(w http.ResponseWriter, r *http.Request) {
	// A visitor may only delete their own videos; the example corpus is not
	// theirs to remove.
	err := s.store.DeleteOwnedVideo(r.Context(), chi.URLParam(r, "id"), ownerKey(r))
	if errors.Is(err, store.ErrNotYours) {
		// Not 403: telling someone their request was forbidden confirms the
		// video exists, which is exactly what they must not learn.
		writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
		return
	}
	if err != nil {
		s.log.Error("delete video", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "could not delete"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

func (s *Server) serveStatic(w http.ResponseWriter, r *http.Request) {
	path := filepath.Join(s.cfg.StaticDir, filepath.Clean(r.URL.Path))
	if info, err := os.Stat(path); err == nil && !info.IsDir() {
		http.ServeFile(w, r, path)
		return
	}
	http.ServeFile(w, r, filepath.Join(s.cfg.StaticDir, "index.html"))
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
