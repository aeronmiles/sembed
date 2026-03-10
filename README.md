# sembed

Minimal, zero-dependency, provider-agnostic semantic embeddings for Go.

## Features

- **`Embedder` interface** -- one method, any provider
- **Built-in providers** -- OpenAI, Voyage AI, Ollama, or any OpenAI-compatible endpoint
- **In-memory index** -- brute-force cosine similarity search, concurrency-safe
- **JSON persistence** -- `Save`/`Load` via `io.Writer`/`io.Reader`
- **Staleness detection** -- content hashing to know when embeddings need refreshing
- **Zero dependencies** -- only the Go standard library

## Install

```bash
go get github.com/aeronmiles/sembed
```

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/aeronmiles/sembed"
)

func main() {
	ctx := context.Background()

	// Create a Voyage AI embedder
	embedder := sembed.VoyageAI(os.Getenv("VOYAGE_API_KEY"), "voyage-3-lite")

	// Embed some documents
	texts := []string{
		"the cat sat on the mat",
		"a dog played in the park",
		"quantum computing breaks encryption",
	}
	vectors, _ := embedder.Embed(ctx, texts)

	// Build an index
	idx := sembed.NewIndex()
	for i, text := range texts {
		idx.Add(sembed.Document{
			ID:          fmt.Sprintf("doc-%d", i),
			Content:     text,
			ContentHash: sembed.ContentHash(text),
			Vector:      vectors[i],
		})
	}

	// Search
	query, _ := embedder.Embed(ctx, []string{"feline sitting"})
	results := idx.Search(query[0], 2)
	for _, r := range results {
		fmt.Printf("%.4f  %s\n", r.Score, r.Document.Content)
	}
}
```

## Providers

### OpenAI

```go
embedder := sembed.OpenAI(apiKey, "text-embedding-3-small")

// With reduced dimensions
embedder = sembed.OpenAI(apiKey, "text-embedding-3-large", sembed.WithDimensions(512))
```

### Voyage AI

```go
embedder := sembed.VoyageAI(apiKey, "voyage-3-lite")

// With input type hint for asymmetric search
embedder = sembed.VoyageAI(apiKey, "voyage-3-lite", sembed.WithInputType("document"))
queryEmb := sembed.VoyageAI(apiKey, "voyage-3-lite", sembed.WithInputType("query"))
```

### Ollama

```go
// Defaults to http://localhost:11434
embedder := sembed.NewOllama(sembed.OllamaConfig{
	Model: "nomic-embed-text",
})

// Custom host
embedder = sembed.NewOllama(sembed.OllamaConfig{
	BaseURL: "http://gpu-server:11434",
	Model:   "mxbai-embed-large",
})
```

### Custom OpenAI-compatible endpoint

Works with vLLM, LiteLLM, Azure, or any endpoint that speaks the OpenAI embeddings protocol.

```go
embedder := sembed.NewOpenAICompatible(sembed.OpenAIConfig{
	BaseURL:    "https://my-vllm-server/v1",
	APIKey:     "token",
	Model:      "BAAI/bge-small-en-v1.5",
	Dimensions: 384,
})
```

## Index

### Add and search

```go
idx := sembed.NewIndex()

// Add documents (upsert semantics -- same ID overwrites)
idx.Add(sembed.Document{
	ID:          "task-1",
	Content:     "Fix login bug",
	ContentHash: sembed.ContentHash("Fix login bug"),
	Vector:      vec,
	Metadata:    map[string]string{"status": "todo"},
})

// Top-k search by cosine similarity
results := idx.Search(queryVec, 5)

// Retrieve by ID
doc, ok := idx.Get("task-1")

// Remove
idx.Remove("task-1")

// Count and list
n := idx.Len()
all := idx.All()
```

### Save and Load

```go
// Save to file
f, _ := os.Create("index.json")
idx.Save(f)
f.Close()

// Load from file (merges into existing index)
f, _ = os.Open("index.json")
idx.Load(f)
f.Close()
```

### Stale detection

Detect which documents need re-embedding after content changes.

```go
// Build a map of current content hashes
current := map[string]string{
	"task-1": sembed.ContentHash("Fix login bug (updated)"),
	"task-2": sembed.ContentHash("Add search feature"),
}

// Returns IDs whose stored hash differs from current
staleIDs := idx.Stale(current)
// Re-embed only the stale documents
```

## Bring Your Own Embedder

Implement the `Embedder` interface to use any provider:

```go
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([]Vector, error)
}
```

Example with a hypothetical local model:

```go
type localEmbedder struct {
	model *someml.Model
}

func (e *localEmbedder) Embed(ctx context.Context, texts []string) ([]sembed.Vector, error) {
	vectors := make([]sembed.Vector, len(texts))
	for i, text := range texts {
		vectors[i] = sembed.Vector(e.model.Encode(text))
	}
	return vectors, nil
}
```

## Vector utilities

```go
sembed.CosineSimilarity(a, b) // cosine similarity between two vectors
sembed.Dot(a, b)              // dot product
sembed.Magnitude(v)           // L2 norm
sembed.Normalize(v)           // unit vector
sembed.ContentHash(text)      // SHA-256 hex digest
```

## License

MIT
