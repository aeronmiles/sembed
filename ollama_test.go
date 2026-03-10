package sembed

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOllama_SuccessfulEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/api/embed" {
			t.Errorf("expected /api/embed path, got %s", r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("expected Content-Type application/json, got %s", ct)
		}

		var reqBody ollamaRequest
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &reqBody)

		if reqBody.Model != "nomic-embed-text" {
			t.Errorf("expected model 'nomic-embed-text', got %s", reqBody.Model)
		}

		resp := ollamaResponse{
			Model: "nomic-embed-text",
			Embeddings: [][]float32{
				{0.1, 0.2, 0.3},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewOllama(OllamaConfig{
		BaseURL: server.URL,
		Model:   "nomic-embed-text",
	})

	vectors, err := client.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vectors) != 1 {
		t.Fatalf("expected 1 vector, got %d", len(vectors))
	}
	if vectors[0][0] != 0.1 || vectors[0][1] != 0.2 || vectors[0][2] != 0.3 {
		t.Errorf("vector mismatch: %v", vectors[0])
	}
}

func TestOllama_DefaultBaseURL(t *testing.T) {
	client := NewOllama(OllamaConfig{
		Model: "nomic-embed-text",
	})

	oc, ok := client.(*ollamaClient)
	if !ok {
		t.Fatal("expected *ollamaClient")
	}
	if oc.cfg.BaseURL != "http://localhost:11434" {
		t.Errorf("expected default base URL http://localhost:11434, got %s", oc.cfg.BaseURL)
	}
}

func TestOllama_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "model not found"}`))
	}))
	defer server.Close()

	client := NewOllama(OllamaConfig{
		BaseURL: server.URL,
		Model:   "nonexistent-model",
	})

	_, err := client.Embed(context.Background(), []string{"hello"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	embedErr, ok := err.(*EmbedError)
	if !ok {
		t.Fatalf("expected *EmbedError, got %T: %v", err, err)
	}
	if embedErr.StatusCode != 500 {
		t.Errorf("expected status 500, got %d", embedErr.StatusCode)
	}
	if embedErr.Body == "" {
		t.Error("expected non-empty error body")
	}
}

func TestOllama_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request should not have reached the server")
	}))
	defer server.Close()

	client := NewOllama(OllamaConfig{
		BaseURL: server.URL,
		Model:   "nomic-embed-text",
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	_, err := client.Embed(ctx, []string{"hello"})
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}

func TestOllama_MultipleTexts(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody ollamaRequest
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &reqBody)

		if len(reqBody.Input) != 3 {
			t.Errorf("expected 3 input texts, got %d", len(reqBody.Input))
		}

		resp := ollamaResponse{
			Model: "nomic-embed-text",
			Embeddings: [][]float32{
				{0.1, 0.2},
				{0.3, 0.4},
				{0.5, 0.6},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewOllama(OllamaConfig{
		BaseURL: server.URL,
		Model:   "nomic-embed-text",
	})

	vectors, err := client.Embed(context.Background(), []string{"one", "two", "three"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vectors) != 3 {
		t.Fatalf("expected 3 vectors, got %d", len(vectors))
	}
	// Verify each vector matches the expected values in order.
	expected := [][]float32{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}
	for i, vec := range vectors {
		if len(vec) != 2 {
			t.Errorf("vector %d: expected length 2, got %d", i, len(vec))
			continue
		}
		if vec[0] != expected[i][0] || vec[1] != expected[i][1] {
			t.Errorf("vector %d: expected %v, got %v", i, expected[i], vec)
		}
	}
}
