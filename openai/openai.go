package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"

	"github.com/aeronmiles/sembed"
)

// Config configures an OpenAI-compatible embeddings provider.
type Config struct {
	// BaseURL is the API base (e.g. "https://api.openai.com/v1").
	// Must not include the /embeddings path.
	BaseURL string

	// APIKey for authentication via Bearer token.
	APIKey string

	// Model name (e.g. "text-embedding-3-small", "voyage-4-lite").
	Model string

	// Dimensions is the desired output dimensions (0 = provider default).
	// Not all providers support this (e.g. Voyage AI does not).
	Dimensions int

	// InputType is an optional type hint ("query" or "document").
	// Used by Voyage AI; ignored by OpenAI.
	InputType string

	// SkipEncodingFormat suppresses the encoding_format field in requests.
	// Set for providers that don't support it (e.g. Voyage AI).
	SkipEncodingFormat bool

	// SkipDimensions suppresses the dimensions field in requests.
	// Set for providers that don't support it (e.g. Voyage AI).
	SkipDimensions bool
}

// client implements sembed.Embedder for OpenAI-compatible APIs.
type client struct {
	cfg        Config
	httpClient *http.Client
}

// request is the JSON request body for the embeddings endpoint.
type request struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
	Dimensions     int      `json:"dimensions,omitempty"`
	InputType      string   `json:"input_type,omitempty"`
}

// response is the JSON response from the embeddings endpoint.
type response struct {
	Data []embedding `json:"data"`
}

// embedding is a single embedding in the response.
type embedding struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// NewCompatible creates an Embedder for any OpenAI-compatible API.
// The config must include BaseURL, APIKey, and Model at minimum.
func NewCompatible(cfg Config) sembed.Embedder {
	return &client{
		cfg:        cfg,
		httpClient: &http.Client{},
	}
}

// VoyageAI creates an Embedder for Voyage AI.
// Base URL: https://api.voyageai.com/v1
func VoyageAI(apiKey, model string, opts ...sembed.Option) sembed.Embedder {
	o := sembed.ApplyOptions(opts)
	return NewCompatible(Config{
		BaseURL:            "https://api.voyageai.com/v1",
		APIKey:             apiKey,
		Model:              model,
		InputType:          o.InputType,
		SkipEncodingFormat: true,
		SkipDimensions:     true,
	})
}

// New creates an Embedder for OpenAI.
// Base URL: https://api.openai.com/v1
func New(apiKey, model string, opts ...sembed.Option) sembed.Embedder {
	o := sembed.ApplyOptions(opts)
	return NewCompatible(Config{
		BaseURL:    "https://api.openai.com/v1",
		APIKey:     apiKey,
		Model:      model,
		Dimensions: o.Dimensions,
		InputType:  o.InputType,
	})
}

// Embed sends texts to the OpenAI-compatible embeddings endpoint and returns
// the resulting vectors in the same order as the input texts.
func (c *client) Embed(ctx context.Context, texts []string) ([]sembed.Vector, error) {
	reqBody := request{
		Model:     c.cfg.Model,
		Input:     texts,
		InputType: c.cfg.InputType,
	}
	if !c.cfg.SkipEncodingFormat {
		reqBody.EncodingFormat = "float"
	}
	if !c.cfg.SkipDimensions {
		reqBody.Dimensions = c.cfg.Dimensions
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := c.cfg.BaseURL + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)

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

	// Sort by index to ensure vectors match input order.
	sort.Slice(result.Data, func(i, j int) bool {
		return result.Data[i].Index < result.Data[j].Index
	})

	vectors := make([]sembed.Vector, len(result.Data))
	for i, d := range result.Data {
		vectors[i] = sembed.Vector(d.Embedding)
	}

	return vectors, nil
}
