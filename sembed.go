package sembed

import "context"

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

// Option configures provider behavior.
type Option func(*options)

type options struct {
	Dimensions int
	InputType  string
}

// WithDimensions sets the output embedding dimensions.
func WithDimensions(d int) Option {
	return func(o *options) { o.Dimensions = d }
}

// WithInputType sets the input type hint (e.g. "query", "document" for Voyage AI).
func WithInputType(t string) Option {
	return func(o *options) { o.InputType = t }
}

// applyOptions merges variadic options into an options struct.
func applyOptions(opts []Option) options {
	var o options
	for _, fn := range opts {
		fn(&o)
	}
	return o
}
