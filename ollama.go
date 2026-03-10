package sembed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// OllamaConfig configures the Ollama embeddings provider.
type OllamaConfig struct {
	// BaseURL defaults to "http://localhost:11434" if empty.
	BaseURL string

	// Model name (e.g. "nomic-embed-text", "mxbai-embed-large").
	Model string
}

// ollamaClient implements Embedder for a local Ollama instance.
type ollamaClient struct {
	cfg    OllamaConfig
	client *http.Client
}

// ollamaRequest is the JSON request body for the Ollama embed endpoint.
type ollamaRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// ollamaResponse is the JSON response from the Ollama embed endpoint.
type ollamaResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}

// NewOllama creates an Embedder for a local Ollama instance.
// If BaseURL is empty, it defaults to "http://localhost:11434".
func NewOllama(cfg OllamaConfig) Embedder {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}
	return &ollamaClient{
		cfg:    cfg,
		client: &http.Client{},
	}
}

// Embed sends texts to the Ollama embed endpoint and returns the resulting
// vectors in the same order as the input texts.
func (c *ollamaClient) Embed(ctx context.Context, texts []string) ([]Vector, error) {
	reqBody := ollamaRequest{
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

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, &EmbedError{
			StatusCode: resp.StatusCode,
			Body:       string(respBody),
		}
	}

	var result ollamaResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	vectors := make([]Vector, len(result.Embeddings))
	for i, emb := range result.Embeddings {
		vectors[i] = Vector(emb)
	}

	return vectors, nil
}
