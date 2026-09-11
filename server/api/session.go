package api

import (
	"crypto/rand"
	"encoding/base64"
	"net/http"
	"time"
)

// A visitor gets an opaque session id in a cookie so they can add videos and
// find them again, without an account. It identifies a browser, nothing more:
// no personal data is stored against it, and it expires on its own.
const (
	sessionCookie = "vs_session"
	sessionTTL    = 30 * 24 * time.Hour
	// Session videos stop being searchable after this, so an abandoned upload
	// does not sit in the index forever.
	videoTTL = 7 * 24 * time.Hour
)

type sessionKey struct{}

func newSessionID() string {
	buf := make([]byte, 18)
	if _, err := rand.Read(buf); err != nil {
		// crypto/rand failing is not recoverable and must not silently produce
		// a guessable id.
		panic("session: " + err.Error())
	}
	return base64.RawURLEncoding.EncodeToString(buf)
}

// withSession ensures every request carries a session id, minting one the first
// time a browser arrives.
func (s *Server) withSession(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := ""
		if c, err := r.Cookie(sessionCookie); err == nil && len(c.Value) >= 16 {
			id = c.Value
		}
		if id == "" {
			id = newSessionID()
			http.SetCookie(w, &http.Cookie{
				Name:     sessionCookie,
				Value:    id,
				Path:     "/",
				MaxAge:   int(sessionTTL.Seconds()),
				HttpOnly: true, // the frontend never needs to read it
				SameSite: http.SameSiteLaxMode,
				Secure:   r.TLS != nil,
			})
		}
		next.ServeHTTP(w, r.WithContext(contextWithSession(r, id)))
	})
}

func sessionFrom(r *http.Request) string {
	if id, ok := r.Context().Value(sessionKey{}).(string); ok {
		return id
	}
	return ""
}
