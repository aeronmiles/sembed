package sembed

import (
	"crypto/sha256"
	"encoding/hex"
)

// ContentHash returns the SHA-256 hex digest of content.
// Suitable for staleness detection in embedding indexes.
func ContentHash(content string) string {
	h := sha256.Sum256([]byte(content))
	return hex.EncodeToString(h[:])
}
