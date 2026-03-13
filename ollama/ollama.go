package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/aeronmiles/sembed"
)

// Config configures the Ollama embeddings provider.
type Config struct {
	// BaseURL defaults to "http://localhost:11434" if empty.
	BaseURL string

	// Model name (e.g. "nomic-embed-text", "mxbai-embed-large").
	Model string
}

// client implements sembed.Embedder for a local Ollama instance.
type client struct {
	cfg        Config
	httpClient *http.Client
}

// request is the JSON request body for the Ollama embed endpoint.
type request struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// response is the JSON response from the Ollama embed endpoint.
type response struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}

// New creates an Embedder for a local Ollama instance.
// If BaseURL is empty, it defaults to "http://localhost:11434".
func New(cfg Config) sembed.Embedder {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}
	return &client{
		cfg:        cfg,
		httpClient: &http.Client{},
	}
}

// Embed sends texts to the Ollama embed endpoint and returns the resulting
// vectors in the same order as the input texts.
func (c *client) Embed(ctx context.Context, texts []string) ([]sembed.Vector, error) {
	reqBody := request{
		Model: c.cfg.Model,
		Input: texts,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := c.cfg.BaseURL + "/api/embed"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, &sembed.EmbedError{
			StatusCode: resp.StatusCode,
			Body:       string(respBody),
		}
	}

	var result response
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	vectors := make([]sembed.Vector, len(result.Embeddings))
	for i, emb := range result.Embeddings {
		vectors[i] = sembed.Vector(emb)
	}

	return vectors, nil
}
