package sembed

import (
	"encoding/json"
	"errors"
	"io"
	"sort"
	"sync"
)

// Index is a flat in-memory vector index with brute-force cosine search.
// It is safe for concurrent use.
type Index struct {
	mu   sync.RWMutex
	docs map[string]Document // keyed by ID
}

// NewIndex creates an empty index.
func NewIndex() *Index {
	return &Index{
		docs: make(map[string]Document),
	}
}

// Add inserts or updates documents by ID (upsert semantics).
// Returns an error if any document has an empty ID.
func (idx *Index) Add(docs ...Document) error {
	for _, d := range docs {
		if d.ID == "" {
			return errors.New("sembed: document ID must not be empty")
		}
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	for _, d := range docs {
		idx.docs[d.ID] = d
	}
	return nil
}

// Remove deletes documents by ID. Missing IDs are silently ignored.
func (idx *Index) Remove(ids ...string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	for _, id := range ids {
		delete(idx.docs, id)
	}
}

// Get returns a document by ID and whether it was found.
func (idx *Index) Get(id string) (Document, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	doc, ok := idx.docs[id]
	return doc, ok
}

// Search returns the top-k documents most similar to the query vector,
// sorted by descending cosine similarity. If k <= 0 or the index is empty,
// returns nil. If k > Len(), returns all documents sorted.
func (idx *Index) Search(query Vector, k int) []Result {
	if k <= 0 {
		return nil
	}
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if len(idx.docs) == 0 {
		return nil
	}
	results := make([]Result, 0, len(idx.docs))
	for _, doc := range idx.docs {
		score := CosineSimilarity(query, doc.Vector)
		results = append(results, Result{
			Document: doc,
			Score:    score,
		})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if k > len(results) {
		k = len(results)
	}
	return results[:k]
}

// Stale returns document IDs where the stored ContentHash differs from
// the corresponding value in current. IDs present in current but not in
// the index are ignored. IDs in the index but not in current are considered stale.
func (idx *Index) Stale(current map[string]string) []string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	var stale []string
	for id, doc := range idx.docs {
		hash, ok := current[id]
		if !ok || hash != doc.ContentHash {
			stale = append(stale, id)
		}
	}
	return stale
}

// All returns a copy of all documents in the index (no guaranteed order).
func (idx *Index) All() []Document {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	docs := make([]Document, 0, len(idx.docs))
	for _, doc := range idx.docs {
		docs = append(docs, doc)
	}
	return docs
}

// Len returns the number of documents in the index.
func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.docs)
}

// indexJSON is the serialization wrapper for Save/Load.
type indexJSON struct {
	Documents []Document `json:"documents"`
}

// Save writes the index as JSON to w.
func (idx *Index) Save(w io.Writer) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	docs := make([]Document, 0, len(idx.docs))
	for _, doc := range idx.docs {
		docs = append(docs, doc)
	}
	data, err := json.MarshalIndent(indexJSON{Documents: docs}, "", "  ")
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

// Load reads JSON from r and merges documents into the index (upsert).
func (idx *Index) Load(r io.Reader) error {
	var wrapper indexJSON
	if err := json.NewDecoder(r).Decode(&wrapper); err != nil {
		return err
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	for _, doc := range wrapper.Documents {
		idx.docs[doc.ID] = doc
	}
	return nil
}
