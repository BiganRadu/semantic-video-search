package api

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	"videosearch/server/store"

	"videosearch/server/auth"
)

// An account is optional. The anonymous session from session.go still works and
// still expires; signing in only changes which key owns a visitor's videos.
const (
	authCookie = "vs_auth"
	authTTL    = 30 * 24 * time.Hour
)

type userKey struct{}

// ownerKey is the value stored in videos.owner, and the single definition of
// "whose corpus is this". It is namespaced so that an anonymous session id can
// never collide with an account id, and so the column says what it holds when
// read by a human.
//
// An empty string means the shared example corpus, which belongs to nobody.
func ownerKey(r *http.Request) string {
	if u := userFrom(r); u != nil {
		return "user:" + u.ID
	}
	if id := sessionFrom(r); id != "" {
		return "anon:" + id
	}
	return ""
}

func userFrom(r *http.Request) *store.User {
	if u, ok := r.Context().Value(userKey{}).(*store.User); ok {
		return u
	}
	return nil
}

// newToken mints a session cookie value. 32 bytes from crypto/rand: this is a
// bearer credential, so guessing one is signing in as someone else.
func newToken() string {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		panic("auth: " + err.Error())
	}
	return base64.RawURLEncoding.EncodeToString(buf)
}

// tokenHash is what reaches the database. The cookie value never does.
func tokenHash(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}

// withAccount resolves the login cookie into a user, once per request.
//
// A cookie that does not resolve is cleared rather than ignored, so a browser
// holding a revoked or expired session stops sending it instead of retrying
// forever.
func (s *Server) withAccount(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := r.Cookie(authCookie)
		if err != nil || len(c.Value) < 16 {
			next.ServeHTTP(w, r)
			return
		}
		user, err := s.store.UserForSession(r.Context(), tokenHash(c.Value))
		if err != nil {
			if !errors.Is(err, store.ErrNoUser) {
				s.log.Error("resolve session", "err", err)
			}
			s.clearAuthCookie(w, r)
			next.ServeHTTP(w, r)
			return
		}
		next.ServeHTTP(w, r.WithContext(contextWithUser(r, user)))
	})
}

func (s *Server) setAuthCookie(w http.ResponseWriter, r *http.Request, token string) {
	http.SetCookie(w, &http.Cookie{
		Name:     authCookie,
		Value:    token,
		Path:     "/",
		MaxAge:   int(authTTL.Seconds()),
		HttpOnly: true, // script must never be able to read a session token
		SameSite: http.SameSiteLaxMode,
		Secure:   r.TLS != nil,
	})
}

func (s *Server) clearAuthCookie(w http.ResponseWriter, r *http.Request) {
	http.SetCookie(w, &http.Cookie{
		Name: authCookie, Value: "", Path: "/", MaxAge: -1,
		HttpOnly: true, SameSite: http.SameSiteLaxMode, Secure: r.TLS != nil,
	})
}

// --- endpoints -------------------------------------------------------------

type credentials struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

// readCredentials decodes the body with a size limit. Without one, an unbounded
// JSON body on an unauthenticated endpoint is a free way to spend the server's
// memory.
func readCredentials(w http.ResponseWriter, r *http.Request) (credentials, bool) {
	var c credentials
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<10))
	if err := dec.Decode(&c); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "expected JSON with email and password"})
		return c, false
	}
	return c, true
}

func (s *Server) apiRegister(w http.ResponseWriter, r *http.Request) {
	c, ok := readCredentials(w, r)
	if !ok {
		return
	}
	email, err := auth.NormalizeEmail(c.Email)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if err := auth.CheckPassword(c.Password); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	hash, err := auth.HashPassword(c.Password)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "could not create the account"})
		return
	}

	user, err := s.store.CreateUser(r.Context(), newSessionID(), email, hash)
	if errors.Is(err, store.ErrEmailTaken) {
		writeJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
		return
	}
	if err != nil {
		s.log.Error("create user", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "could not create the account"})
		return
	}
	s.signIn(w, r, user)
}

func (s *Server) apiLogin(w http.ResponseWriter, r *http.Request) {
	c, ok := readCredentials(w, r)
	if !ok {
		return
	}
	if !s.logins.allow(clientIP(r)) {
		writeJSON(w, http.StatusTooManyRequests,
			map[string]string{"error": "too many sign-in attempts, wait a minute"})
		return
	}

	email, err := auth.NormalizeEmail(c.Email)
	if err != nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": auth.ErrBadPassword.Error()})
		return
	}

	user, hash, err := s.store.CredentialsFor(r.Context(), email)
	if err != nil {
		if !errors.Is(err, store.ErrNoUser) {
			s.log.Error("lookup user", "err", err)
		}
		// Hash anyway. Returning early for an unknown address would make the
		// response time an oracle for which addresses are registered.
		_, _ = auth.HashPassword(c.Password)
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": auth.ErrBadPassword.Error()})
		return
	}
	if err := auth.VerifyPassword(hash, c.Password); err != nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": auth.ErrBadPassword.Error()})
		return
	}
	s.signIn(w, r, user)
}

// signIn is the one place a session is created, so the cookie, the row and the
// video hand-over cannot drift apart between register and login.
func (s *Server) signIn(w http.ResponseWriter, r *http.Request, user *store.User) {
	token := newToken()
	if err := s.store.StartSession(r.Context(), tokenHash(token), user.ID,
		r.UserAgent(), time.Now().Add(authTTL)); err != nil {
		s.log.Error("start session", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "could not sign in"})
		return
	}

	// Anything this browser added anonymously becomes theirs, and stops
	// expiring. Losing an upload because you made an account afterwards would
	// be a surprising way to be punished for signing up.
	claimed, err := s.store.ClaimVideos(r.Context(), "anon:"+sessionFrom(r), "user:"+user.ID)
	if err != nil {
		// Not fatal: the sign-in itself succeeded, and the videos are still
		// under the anonymous session until it expires.
		s.log.Error("claim videos", "err", err, "user", user.ID)
	}

	s.setAuthCookie(w, r, token)
	writeJSON(w, http.StatusOK, map[string]any{"user": user, "claimed_videos": claimed})
}

func (s *Server) apiLogout(w http.ResponseWriter, r *http.Request) {
	if c, err := r.Cookie(authCookie); err == nil && c.Value != "" {
		if err := s.store.EndSession(r.Context(), tokenHash(c.Value)); err != nil {
			s.log.Error("end session", "err", err)
		}
	}
	s.clearAuthCookie(w, r)
	writeJSON(w, http.StatusOK, map[string]any{"user": nil})
}

// apiMe is how the frontend learns whether it is signed in. It is deliberately
// 200-with-null rather than 401: not being signed in is a normal state, not an
// error, and the console should not fill with red on every page load.
func (s *Server) apiMe(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"user": userFrom(r)})
}

// --- sign-in throttle ------------------------------------------------------

// limiter is a fixed-window counter per client address. In-process and
// therefore per-instance, which is the right size for a single binary: it makes
// online password guessing expensive without adding Redis to the deployment.
type limiter struct {
	mu      sync.Mutex
	hits    map[string]int
	window  time.Time
	perMin  int
	maxKeys int
}

func newLimiter(perMin int) *limiter {
	return &limiter{hits: map[string]int{}, window: time.Now(), perMin: perMin, maxKeys: 10_000}
}

func (l *limiter) allow(key string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()

	if time.Since(l.window) >= time.Minute {
		l.hits = make(map[string]int, len(l.hits))
		l.window = time.Now()
	}
	// A flood of distinct addresses must not grow the map without bound; the
	// cost of the rare reset is far lower than the cost of the leak.
	if len(l.hits) >= l.maxKeys {
		l.hits = map[string]int{}
	}
	l.hits[key]++
	return l.hits[key] <= l.perMin
}

func clientIP(r *http.Request) string {
	// chi's RealIP middleware has already applied X-Forwarded-For where it is
	// trusted, so RemoteAddr is the address to throttle on.
	host := r.RemoteAddr
	if i := strings.LastIndex(host, ":"); i > 0 {
		host = host[:i]
	}
	return host
}
