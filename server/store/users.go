package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// ErrEmailTaken is returned rather than the raw constraint violation, so the
// handler does not have to know the name of an index.
var ErrEmailTaken = errors.New("that email is already registered")

// ErrNoUser covers both "no such account" and "session expired". Callers turn
// it into the same response either way.
var ErrNoUser = errors.New("no such user")

type User struct {
	ID        string    `json:"id"`
	Email     string    `json:"email"`
	CreatedAt time.Time `json:"created_at"`
}

// CreateUser stores an account. The password must already be hashed: this
// package never sees a plaintext one, which is the point of the split.
func (s *Store) CreateUser(ctx context.Context, id, email, passwordHash string) (*User, error) {
	var u User
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO users (id, email, password_hash)
		VALUES ($1, $2, $3)
		RETURNING id, email, created_at`, id, email, passwordHash).
		Scan(&u.ID, &u.Email, &u.CreatedAt)

	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23505" { // unique_violation
		return nil, ErrEmailTaken
	}
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// CredentialsFor returns the stored hash for an address, for verification.
//
// It returns ErrNoUser for an unknown address, and the caller is expected to
// hash the supplied password anyway before answering -- otherwise the response
// time says whether the address is registered.
func (s *Store) CredentialsFor(ctx context.Context, email string) (*User, string, error) {
	var u User
	var hash string
	err := s.Pool.QueryRow(ctx, `
		SELECT id, email, created_at, password_hash
		FROM users WHERE lower(email) = lower($1)`, email).
		Scan(&u.ID, &u.Email, &u.CreatedAt, &hash)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, "", ErrNoUser
	}
	if err != nil {
		return nil, "", err
	}
	return &u, hash, nil
}

// StartSession records a login. tokenHash is the digest of the cookie value;
// the value itself is never stored, so a database dump is not a set of logins.
func (s *Store) StartSession(ctx context.Context, tokenHash, userID, userAgent string, expires time.Time) error {
	_, err := s.Pool.Exec(ctx, `
		INSERT INTO auth_sessions (token_hash, user_id, expires_at, user_agent)
		VALUES ($1, $2, $3, nullif($4, ''))`, tokenHash, userID, expires, userAgent)
	return err
}

// UserForSession resolves a login cookie, and is the only path by which a
// request becomes authenticated.
func (s *Store) UserForSession(ctx context.Context, tokenHash string) (*User, error) {
	var u User
	err := s.Pool.QueryRow(ctx, `
		SELECT u.id, u.email, u.created_at
		FROM auth_sessions a JOIN users u ON u.id = a.user_id
		WHERE a.token_hash = $1 AND a.expires_at > now()`, tokenHash).
		Scan(&u.ID, &u.Email, &u.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNoUser
	}
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// EndSession signs one session out. Sessions are rows rather than signed
// cookies precisely so that this works: logging out has to actually revoke,
// not merely ask the browser to forget.
func (s *Store) EndSession(ctx context.Context, tokenHash string) error {
	_, err := s.Pool.Exec(ctx, `DELETE FROM auth_sessions WHERE token_hash = $1`, tokenHash)
	return err
}

// SweepExpiredSessions drops logins that have run out, alongside the video
// sweep. Nothing depends on it for correctness -- UserForSession already checks
// the expiry -- it just stops the table growing forever.
func (s *Store) SweepExpiredSessions(ctx context.Context) (int64, error) {
	tag, err := s.Pool.Exec(ctx, `DELETE FROM auth_sessions WHERE expires_at <= now()`)
	if err != nil {
		return 0, err
	}
	return tag.RowsAffected(), nil
}

// ClaimVideos moves whatever an anonymous browser added onto the account that
// just signed in, and clears the expiry: a video kept by an account is kept
// until its owner deletes it.
//
// This runs on sign-in and sign-up both, so the common path -- add a video,
// like it, make an account to keep it -- does not silently lose the video. It
// can only ever move videos owned by the session making the request, so it
// cannot be used to take someone else's.
func (s *Store) ClaimVideos(ctx context.Context, fromOwner, toOwner string) (int64, error) {
	if fromOwner == "" || toOwner == "" || fromOwner == toOwner {
		return 0, nil
	}
	tag, err := s.Pool.Exec(ctx, `
		UPDATE videos SET owner = $2, expires_at = NULL
		WHERE owner = $1`, fromOwner, toOwner)
	if err != nil {
		return 0, fmt.Errorf("claim videos: %w", err)
	}
	return tag.RowsAffected(), nil
}
