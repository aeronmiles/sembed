package sembed

import (
	"regexp"
	"testing"
)

func TestContentHash_Deterministic(t *testing.T) {
	a := ContentHash("hello world")
	b := ContentHash("hello world")
	if a != b {
		t.Errorf("same content produced different hashes: %q vs %q", a, b)
	}
}

func TestContentHash_DifferentContent(t *testing.T) {
	a := ContentHash("hello")
	b := ContentHash("world")
	if a == b {
		t.Errorf("different content produced same hash: %q", a)
	}
}

func TestContentHash_EmptyString(t *testing.T) {
	h := ContentHash("")
	if h == "" {
		t.Error("empty string produced empty hash")
	}
	if len(h) != 64 {
		t.Errorf("hash length: got %d, want 64", len(h))
	}
}

func TestContentHash_HexFormat(t *testing.T) {
	h := ContentHash("test content")
	matched, err := regexp.MatchString("^[0-9a-f]{64}$", h)
	if err != nil {
		t.Fatal(err)
	}
	if !matched {
		t.Errorf("hash is not 64-char hex: %q", h)
	}
}
