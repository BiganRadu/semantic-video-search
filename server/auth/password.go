// Package auth holds password hashing and the rules about what makes an
// acceptable credential. It knows nothing about HTTP or the database, which is
// what keeps the rules testable on their own -- see password_test.go.
package auth

import (
	"crypto/pbkdf2"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"fmt"
	"net/mail"
	"strconv"
	"strings"
	"unicode/utf8"
)

// PBKDF2-HMAC-SHA256. Not the strongest choice available -- argon2id is better
// against GPU attack -- but it is in the standard library as of Go 1.24, and a
// correct stdlib KDF beats a dependency that has to be fetched, pinned and
// trusted. The iteration count is the OWASP 2023 figure for this construction.
const (
	iterations = 600_000
	saltLen    = 16
	keyLen     = 32
	scheme     = "pbkdf2-sha256"
)

// Passwords are capped because every byte is hashed: without a limit, a
// multi-megabyte password is a free way to pin a CPU core.
const (
	MinPasswordLen = 8
	MaxPasswordLen = 1024
	MaxEmailLen    = 254 // RFC 5321 limit on a forward path
)

var (
	ErrBadEmail    = errors.New("that does not look like an email address")
	ErrShortPass   = fmt.Errorf("password must be at least %d characters", MinPasswordLen)
	ErrLongPass    = fmt.Errorf("password must be at most %d characters", MaxPasswordLen)
	ErrBadPassword = errors.New("wrong email or password")
)

// NormalizeEmail validates an address and puts it in the form it is stored and
// compared in. Only the case is folded: the local part of an address is
// case-sensitive by the RFC, but no mail provider in practice treats it that
// way, and allowing Bob@x and bob@x to be two accounts is a takeover vector.
func NormalizeEmail(raw string) (string, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" || len(trimmed) > MaxEmailLen {
		return "", ErrBadEmail
	}
	addr, err := mail.ParseAddress(trimmed)
	if err != nil || addr.Address != trimmed || !strings.Contains(trimmed, ".") {
		// ParseAddress accepts `Name <a@b>`; only a bare address is an account.
		return "", ErrBadEmail
	}
	return strings.ToLower(addr.Address), nil
}

// CheckPassword rejects a password before it is ever hashed or stored.
func CheckPassword(pw string) error {
	// Counted in runes, so a passphrase in a non-Latin script is not punished
	// for its bytes.
	if utf8.RuneCountInString(pw) < MinPasswordLen {
		return ErrShortPass
	}
	if len(pw) > MaxPasswordLen {
		return ErrLongPass
	}
	return nil
}

// HashPassword returns a self-describing digest: scheme, cost, salt and key.
// Everything needed to verify it later is in the string, so the cost can be
// raised for new passwords without invalidating existing ones.
func HashPassword(pw string) (string, error) {
	salt := make([]byte, saltLen)
	if _, err := rand.Read(salt); err != nil {
		return "", err
	}
	key, err := pbkdf2.Key(sha256.New, pw, salt, iterations, keyLen)
	if err != nil {
		return "", err
	}
	return strings.Join([]string{
		scheme,
		strconv.Itoa(iterations),
		base64.RawStdEncoding.EncodeToString(salt),
		base64.RawStdEncoding.EncodeToString(key),
	}, "$"), nil
}

// VerifyPassword reports whether pw produced the stored digest.
//
// Every failure returns the same error. A caller must not be able to tell a
// malformed digest from an unknown scheme from a wrong password, because the
// difference would say something about the account.
func VerifyPassword(encoded, pw string) error {
	parts := strings.Split(encoded, "$")
	if len(parts) != 4 || parts[0] != scheme {
		return ErrBadPassword
	}
	iter, err := strconv.Atoi(parts[1])
	if err != nil || iter <= 0 || iter > 10_000_000 {
		return ErrBadPassword
	}
	salt, err := base64.RawStdEncoding.DecodeString(parts[2])
	if err != nil {
		return ErrBadPassword
	}
	want, err := base64.RawStdEncoding.DecodeString(parts[3])
	if err != nil {
		return ErrBadPassword
	}
	got, err := pbkdf2.Key(sha256.New, pw, salt, iter, len(want))
	if err != nil {
		return ErrBadPassword
	}
	// Constant time: a comparison that returns early leaks how much of the
	// digest matched.
	if subtle.ConstantTimeCompare(got, want) != 1 {
		return ErrBadPassword
	}
	return nil
}
