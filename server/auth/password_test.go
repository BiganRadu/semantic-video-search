package auth

import (
	"strings"
	"testing"
)

func TestHashVerifyRoundTrip(t *testing.T) {
	hash, err := HashPassword("correct horse battery")
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyPassword(hash, "correct horse battery"); err != nil {
		t.Fatalf("correct password rejected: %v", err)
	}
	if err := VerifyPassword(hash, "correct horse batterz"); err == nil {
		t.Fatal("wrong password accepted")
	}
}

func TestHashIsSaltedPerPassword(t *testing.T) {
	// Equal passwords must not produce equal digests, or the database tells an
	// attacker which accounts share one.
	a, _ := HashPassword("same password")
	b, _ := HashPassword("same password")
	if a == b {
		t.Fatal("two hashes of the same password are identical: salt is not being used")
	}
}

func TestHashNeverContainsThePassword(t *testing.T) {
	const pw = "sup3rSecretValue"
	hash, _ := HashPassword(pw)
	if strings.Contains(hash, pw) {
		t.Fatal("the stored digest contains the plaintext")
	}
}

func TestVerifyRejectsMalformedDigests(t *testing.T) {
	// Every one of these must fail closed. A digest that fails *open* on a
	// truncated or foreign value is a login bypass.
	for _, bad := range []string{
		"", "$", "not-a-hash", "pbkdf2-sha256$", "pbkdf2-sha256$0$aaaa$bbbb",
		"pbkdf2-sha256$600000$!!!!$bbbb", "pbkdf2-sha256$600000$aaaa$!!!!",
		"scrypt$600000$aaaa$bbbb", "pbkdf2-sha256$600000$aaaa",
		"pbkdf2-sha256$99999999999$aaaa$bbbb",
	} {
		if err := VerifyPassword(bad, "anything"); err == nil {
			t.Fatalf("malformed digest %q was accepted", bad)
		}
	}
}

func TestVerifyOfEmptyPasswordAgainstEmptyHash(t *testing.T) {
	if err := VerifyPassword("", ""); err == nil {
		t.Fatal("empty digest and empty password authenticated")
	}
}

func TestNormalizeEmail(t *testing.T) {
	for raw, want := range map[string]string{
		"Bob@Example.COM":  "bob@example.com",
		"  a.b@c.io  ":     "a.b@c.io",
		"x+tag@mail.co.uk": "x+tag@mail.co.uk",
	} {
		got, err := NormalizeEmail(raw)
		if err != nil || got != want {
			t.Fatalf("NormalizeEmail(%q) = %q, %v; want %q", raw, got, err, want)
		}
	}
}

func TestNormalizeEmailRejects(t *testing.T) {
	// The display-name form matters: "Admin <a@b.com>" must not become an
	// account, and neither must an address with no dot in the domain.
	for _, bad := range []string{
		"", "   ", "nope", "a@b", "Admin <a@b.com>", "a@@b.com",
		strings.Repeat("a", 250) + "@example.com",
	} {
		if got, err := NormalizeEmail(bad); err == nil {
			t.Fatalf("NormalizeEmail(%q) accepted as %q", bad, got)
		}
	}
}

func TestCaseFoldingClosesTheDuplicateAccountHole(t *testing.T) {
	a, _ := NormalizeEmail("Victim@example.com")
	b, _ := NormalizeEmail("victim@example.com")
	if a != b {
		t.Fatal("two cases of one address normalize differently: both could register")
	}
}

func TestCheckPassword(t *testing.T) {
	if err := CheckPassword("short"); err == nil {
		t.Fatal("a 5-character password was accepted")
	}
	if err := CheckPassword(strings.Repeat("a", MaxPasswordLen+1)); err == nil {
		t.Fatal("an oversized password was accepted: hashing it is free CPU for an attacker")
	}
	// Counted in runes, so a short-in-bytes-but-long-in-characters passphrase
	// is judged the same as a Latin one.
	if err := CheckPassword("парольпароль"); err != nil {
		t.Fatalf("a 12-rune non-Latin passphrase was rejected: %v", err)
	}
}
