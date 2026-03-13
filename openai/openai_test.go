package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aeronmiles/sembed"
)

func TestSuccessfulEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/embeddings" {
			t.Errorf("expected /embeddings path, got %s", r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("expected Content-Type application/json, got %s", ct)
		}
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-key" {
			t.Errorf("expected Authorization Bearer test-key, got %s", auth)
		}

		resp := map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": []float32{0.1, 0.2, 0.3}, "index": 0},
				{"embedding": []float32{0.4, 0.5, 0.6}, "index": 1},
			},
			"model": "text-embedding-3-small",
			"usage": map[string]interface{}{"total_tokens": 10},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	embedder := NewCompatible(Config{
		BaseURL: server.URL,
		APIKey:  "test-key",
		Model:   "text-embedding-3-small",
	})

	vectors, err := embedder.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(vectors))
	}
	if vectors[0][0] != 0.1 || vectors[0][1] != 0.2 || vectors[0][2] != 0.3 {
		t.Errorf("vector 0 mismatch: %v", vectors[0])
	}
	if vectors[1][0] != 0.4 || vectors[1][1] != 0.5 || vectors[1][2] != 0.6 {
		t.Errorf("vector 1 mismatch: %v", vectors[1])
	}
}

func TestBatchOrdering(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return data out of order: index 1 before index 0.
		resp := map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": []float32{0.4, 0.5, 0.6}, "index": 1},
				{"embedding": []float32{0.1, 0.2, 0.3}, "index": 0},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	embedder := NewCompatible(Config{
		BaseURL: server.URL,
		APIKey:  "test-key",
		Model:   "test-model",
	})

	vectors, err := embedder.Embed(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(vectors))
	}
	// Index 0 should be first despite being returned second.
	if vectors[0][0] != 0.1 {
		t.Errorf("expected first vector to start with 0.1, got %v", vectors[0][0])
	}
	if vectors[1][0] != 0.4 {
		t.Errorf("expected second vector to start with 0.4, got %v", vectors[1][0])
	}
}

func TestErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error": {"message": "rate limit exceeded"}}`))
	}))
	defer server.Close()

	embedder := NewCompatible(Config{
		BaseURL: server.URL,
		APIKey:  "test-key",
		Model:   "test-model",
	})

	_, err := embedder.Embed(context.Background(), []string{"hello"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	embedErr, ok := err.(*sembed.EmbedError)
	if !ok {
		t.Fatalf("expected *sembed.EmbedError, got %T: %v", err, err)
	}
	if embedErr.StatusCode != 429 {
		t.Errorf("expected status 429, got %d", embedErr.StatusCode)
	}
	if embedErr.Body == "" {
		t.Error("expected non-empty error body")
	}
}

func TestContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request should not have reached the server")
	}))
	defer server.Close()

	embedder := NewCompatible(Config{
		BaseURL: server.URL,
		APIKey:  "test-key",
		Model:   "test-model",
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	_, err := embedder.Embed(ctx, []string{"hello"})
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}

func TestVoyageAI_Constructor(t *testing.T) {
	embedder := VoyageAI("voyage-key", "voyage-4-lite", sembed.WithInputType("document"))

	c, ok := embedder.(*client)
	if !ok {
		t.Fatal("expected *client")
	}
	if c.cfg.BaseURL != "https://api.voyageai.com/v1" {
		t.Errorf("expected Voyage AI base URL, got %s", c.cfg.BaseURL)
	}
	if c.cfg.InputType != "document" {
		t.Errorf("expected input_type 'document', got %s", c.cfg.InputType)
	}
	if c.cfg.APIKey != "voyage-key" {
		t.Errorf("expected API key 'voyage-key', got %s", c.cfg.APIKey)
	}
	if c.cfg.Model != "voyage-4-lite" {
		t.Errorf("expected model 'voyage-4-lite', got %s", c.cfg.Model)
	}
}

func TestNew_Constructor(t *testing.T) {
	embedder := New("openai-key", "text-embedding-3-small")

	c, ok := embedder.(*client)
	if !ok {
		t.Fatal("expected *client")
	}
	if c.cfg.BaseURL != "https://api.openai.com/v1" {
		t.Errorf("expected OpenAI base URL, got %s", c.cfg.BaseURL)
	}
	if c.cfg.APIKey != "openai-key" {
		t.Errorf("expected API key 'openai-key', got %s", c.cfg.APIKey)
	}
	if c.cfg.Model != "text-embedding-3-small" {
		t.Errorf("expected model 'text-embedding-3-small', got %s", c.cfg.Model)
	}
}

func TestWithDimensions(t *testing.T) {
	var capturedBody request

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		resp := map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": []float32{0.1, 0.2}, "index": 0},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	embedder := NewCompatible(Config{
		BaseURL:    server.URL,
		APIKey:     "test-key",
		Model:      "test-model",
		Dimensions: 256,
	})

	_, err := embedder.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if capturedBody.Dimensions != 256 {
		t.Errorf("expected dimensions 256, got %d", capturedBody.Dimensions)
	}
}

func TestWithInputType(t *testing.T) {
	var capturedBody request

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		resp := map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": []float32{0.1}, "index": 0},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	embedder := NewCompatible(Config{
		BaseURL:   server.URL,
		APIKey:    "test-key",
		Model:     "test-model",
		InputType: "query",
	})

	_, err := embedder.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if capturedBody.InputType != "query" {
		t.Errorf("expected input_type 'query', got %s", capturedBody.InputType)
	}
}

func TestDimensionsOmittedWhenZero(t *testing.T) {
	var capturedRaw map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedRaw)

		resp := map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": []float32{0.1}, "index": 0},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	embedder := NewCompatible(Config{
		BaseURL: server.URL,
		APIKey:  "test-key",
		Model:   "test-model",
		// Dimensions: 0 (default)
	})

	_, err := embedder.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, exists := capturedRaw["dimensions"]; exists {
		t.Error("dimensions field should be omitted when zero")
	}
}

func TestInputTypeOmittedWhenEmpty(t *testing.T) {
	var capturedRaw map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedRaw)

		resp := map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": []float32{0.1}, "index": 0},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	embedder := NewCompatible(Config{
		BaseURL: server.URL,
		APIKey:  "test-key",
		Model:   "test-model",
		// InputType: "" (default)
	})

	_, err := embedder.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, exists := capturedRaw["input_type"]; exists {
		t.Error("input_type field should be omitted when empty")
	}
}
