package sembed

import (
	"context"
	"fmt"
)

// Embedder generates vector embeddings from text.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([]Vector, error)
}

// Vector is a dense embedding vector.
type Vector []float32

// Document represents an embedded text with staleness tracking.
type Document struct {
	ID          string            `json:"id"`
	Content     string            `json:"content"`
	ContentHash string            `json:"content_hash"`
	Vector      Vector            `json:"vector"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// Result pairs a document with its similarity score.
type Result struct {
	Document Document `json:"document"`
	Score    float32  `json:"score"`
}

// EmbedError represents an API error from an embedding provider.
type EmbedError struct {
	StatusCode int
	Body       string
}

// Error returns a human-readable error message including the HTTP status code and response body.
func (e *EmbedError) Error() string {
	return fmt.Sprintf("embedding API error (status %d): %s", e.StatusCode, e.Body)
}

// Option configures provider behavior.
type Option func(*Options)

// Options holds resolved provider options.
type Options struct {
	Dimensions int
	InputType  string
}

// WithDimensions sets the output embedding dimensions.
func WithDimensions(d int) Option {
	return func(o *Options) { o.Dimensions = d }
}

// WithInputType sets the input type hint (e.g. "query", "document" for Voyage AI).
func WithInputType(t string) Option {
	return func(o *Options) { o.InputType = t }
}

// ApplyOptions merges variadic options into an Options struct.
func ApplyOptions(opts []Option) Options {
	var o Options
	for _, fn := range opts {
		fn(&o)
	}
	return o
}
