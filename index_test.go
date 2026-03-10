package sembed

import (
	"bytes"
	"math"
	"sort"
	"strings"
	"sync"
	"testing"
)

func TestNewIndex_Empty(t *testing.T) {
	idx := NewIndex()
	if idx.Len() != 0 {
		t.Errorf("NewIndex().Len() = %d, want 0", idx.Len())
	}
}

func TestAdd_Single(t *testing.T) {
	idx := NewIndex()
	doc := Document{ID: "a", Content: "hello", Vector: Vector{1, 0, 0}}
	if err := idx.Add(doc); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if idx.Len() != 1 {
		t.Fatalf("Len = %d, want 1", idx.Len())
	}
	got, ok := idx.Get("a")
	if !ok {
		t.Fatal("Get returned not found")
	}
	if got.Content != "hello" {
		t.Errorf("Content = %q, want %q", got.Content, "hello")
	}
}

func TestAdd_Multiple(t *testing.T) {
	idx := NewIndex()
	docs := []Document{
		{ID: "a", Content: "one", Vector: Vector{1, 0, 0}},
		{ID: "b", Content: "two", Vector: Vector{0, 1, 0}},
		{ID: "c", Content: "three", Vector: Vector{0, 0, 1}},
	}
	if err := idx.Add(docs...); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if idx.Len() != 3 {
		t.Fatalf("Len = %d, want 3", idx.Len())
	}
	for _, d := range docs {
		got, ok := idx.Get(d.ID)
		if !ok {
			t.Errorf("Get(%q) not found", d.ID)
			continue
		}
		if got.Content != d.Content {
			t.Errorf("Get(%q).Content = %q, want %q", d.ID, got.Content, d.Content)
		}
	}
}

func TestAdd_Upsert(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(Document{ID: "a", Content: "v1", Vector: Vector{1, 0, 0}})
	_ = idx.Add(Document{ID: "a", Content: "v2", Vector: Vector{0, 1, 0}})
	if idx.Len() != 1 {
		t.Fatalf("Len = %d, want 1 after upsert", idx.Len())
	}
	got, _ := idx.Get("a")
	if got.Content != "v2" {
		t.Errorf("Content = %q, want %q after upsert", got.Content, "v2")
	}
}

func TestAdd_EmptyID(t *testing.T) {
	idx := NewIndex()
	err := idx.Add(Document{ID: "", Content: "bad"})
	if err == nil {
		t.Fatal("expected error for empty ID, got nil")
	}
	if idx.Len() != 0 {
		t.Errorf("Len = %d, want 0 after failed add", idx.Len())
	}
}

func TestAdd_EmptyID_BatchRollback(t *testing.T) {
	idx := NewIndex()
	err := idx.Add(
		Document{ID: "good", Content: "ok", Vector: Vector{1}},
		Document{ID: "", Content: "bad"},
	)
	if err == nil {
		t.Fatal("expected error for empty ID in batch")
	}
	// The empty-ID validation happens before any insertion, so "good" should not be added.
	if idx.Len() != 0 {
		t.Errorf("Len = %d, want 0 — batch should be rejected atomically", idx.Len())
	}
}

func TestRemove(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(
		Document{ID: "a", Content: "one", Vector: Vector{1, 0}},
		Document{ID: "b", Content: "two", Vector: Vector{0, 1}},
	)
	idx.Remove("a")
	if idx.Len() != 1 {
		t.Fatalf("Len = %d, want 1 after remove", idx.Len())
	}
	if _, ok := idx.Get("a"); ok {
		t.Error("Get(a) should return not-found after remove")
	}
}

func TestRemove_Missing(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(Document{ID: "a", Content: "one", Vector: Vector{1}})
	idx.Remove("nonexistent") // should not panic
	if idx.Len() != 1 {
		t.Errorf("Len = %d, want 1 after removing missing ID", idx.Len())
	}
}

func TestGet_NotFound(t *testing.T) {
	idx := NewIndex()
	_, ok := idx.Get("nope")
	if ok {
		t.Error("Get on empty index should return false")
	}
}

func TestSearch_Ranking(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(
		Document{ID: "best", Content: "exact match", Vector: Vector{1, 0, 0}},
		Document{ID: "mid", Content: "partial match", Vector: Vector{0.7, 0.7, 0}},
		Document{ID: "worst", Content: "orthogonal", Vector: Vector{0, 1, 0}},
	)
	query := Vector{1, 0, 0}
	results := idx.Search(query, 3)
	if len(results) != 3 {
		t.Fatalf("Search returned %d results, want 3", len(results))
	}
	if results[0].Document.ID != "best" {
		t.Errorf("rank 0: got ID %q, want %q", results[0].Document.ID, "best")
	}
	if results[1].Document.ID != "mid" {
		t.Errorf("rank 1: got ID %q, want %q", results[1].Document.ID, "mid")
	}
	if results[2].Document.ID != "worst" {
		t.Errorf("rank 2: got ID %q, want %q", results[2].Document.ID, "worst")
	}
	// Verify score values.
	if !approxEqual(results[0].Score, 1.0) {
		t.Errorf("rank 0 score: got %f, want 1.0", results[0].Score)
	}
	if !approxEqual(results[2].Score, 0.0) {
		t.Errorf("rank 2 score: got %f, want 0.0", results[2].Score)
	}
	// Mid score should be ~0.707.
	expectedMid := float32(0.7 / math.Sqrt(0.7*0.7+0.7*0.7))
	if !approxEqual(results[1].Score, expectedMid) {
		t.Errorf("rank 1 score: got %f, want ~%f", results[1].Score, expectedMid)
	}
}

func TestSearch_TopK(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(
		Document{ID: "a", Vector: Vector{1, 0, 0}},
		Document{ID: "b", Vector: Vector{0.7, 0.7, 0}},
		Document{ID: "c", Vector: Vector{0, 1, 0}},
	)
	results := idx.Search(Vector{1, 0, 0}, 1)
	if len(results) != 1 {
		t.Fatalf("Search(k=1) returned %d results, want 1", len(results))
	}
	if results[0].Document.ID != "a" {
		t.Errorf("top-1 ID = %q, want %q", results[0].Document.ID, "a")
	}
}

func TestSearch_KExceedsLen(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(
		Document{ID: "a", Vector: Vector{1, 0}},
		Document{ID: "b", Vector: Vector{0, 1}},
	)
	results := idx.Search(Vector{1, 0}, 100)
	if len(results) != 2 {
		t.Fatalf("Search(k=100) returned %d results, want 2", len(results))
	}
}

func TestSearch_KZero(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(Document{ID: "a", Vector: Vector{1, 0}})
	results := idx.Search(Vector{1, 0}, 0)
	if results != nil {
		t.Errorf("Search(k=0) = %v, want nil", results)
	}
}

func TestSearch_KNegative(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(Document{ID: "a", Vector: Vector{1, 0}})
	results := idx.Search(Vector{1, 0}, -1)
	if results != nil {
		t.Errorf("Search(k=-1) = %v, want nil", results)
	}
}

func TestSearch_EmptyIndex(t *testing.T) {
	idx := NewIndex()
	results := idx.Search(Vector{1, 0, 0}, 5)
	if results != nil {
		t.Errorf("Search on empty index = %v, want nil", results)
	}
}

func TestStale_MatchingHash(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(Document{ID: "a", ContentHash: "abc123"})
	stale := idx.Stale(map[string]string{"a": "abc123"})
	if len(stale) != 0 {
		t.Errorf("Stale = %v, want empty (hashes match)", stale)
	}
}

func TestStale_DifferentHash(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(Document{ID: "a", ContentHash: "old"})
	stale := idx.Stale(map[string]string{"a": "new"})
	if len(stale) != 1 || stale[0] != "a" {
		t.Errorf("Stale = %v, want [a]", stale)
	}
}

func TestStale_NotInCurrent(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(Document{ID: "a", ContentHash: "abc"})
	stale := idx.Stale(map[string]string{}) // "a" not in current
	if len(stale) != 1 || stale[0] != "a" {
		t.Errorf("Stale = %v, want [a]", stale)
	}
}

func TestStale_AllMatching(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(
		Document{ID: "a", ContentHash: "h1"},
		Document{ID: "b", ContentHash: "h2"},
	)
	stale := idx.Stale(map[string]string{"a": "h1", "b": "h2"})
	if len(stale) != 0 {
		t.Errorf("Stale = %v, want empty", stale)
	}
}

func TestStale_Mixed(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(
		Document{ID: "a", ContentHash: "h1"},
		Document{ID: "b", ContentHash: "h2"},
		Document{ID: "c", ContentHash: "h3"},
	)
	current := map[string]string{
		"a": "h1",       // matches
		"b": "h2_new",   // differs
		// "c" missing → stale
	}
	stale := idx.Stale(current)
	sort.Strings(stale)
	if len(stale) != 2 || stale[0] != "b" || stale[1] != "c" {
		t.Errorf("Stale = %v, want [b c]", stale)
	}
}

func TestSaveLoad_RoundTrip(t *testing.T) {
	idx := NewIndex()
	docs := []Document{
		{ID: "a", Content: "hello", ContentHash: "h1", Vector: Vector{1, 0, 0}, Metadata: map[string]string{"k": "v"}},
		{ID: "b", Content: "world", ContentHash: "h2", Vector: Vector{0, 1, 0}},
	}
	_ = idx.Add(docs...)

	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Verify it's valid JSON with the expected wrapper.
	saved := buf.String()
	if !strings.Contains(saved, `"documents"`) {
		t.Error("saved JSON should contain 'documents' key")
	}

	idx2 := NewIndex()
	if err := idx2.Load(strings.NewReader(saved)); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if idx2.Len() != 2 {
		t.Fatalf("loaded Len = %d, want 2", idx2.Len())
	}

	got, ok := idx2.Get("a")
	if !ok {
		t.Fatal("loaded index missing doc 'a'")
	}
	if got.Content != "hello" {
		t.Errorf("Content = %q, want %q", got.Content, "hello")
	}
	if got.ContentHash != "h1" {
		t.Errorf("ContentHash = %q, want %q", got.ContentHash, "h1")
	}
	if len(got.Vector) != 3 || got.Vector[0] != 1 {
		t.Errorf("Vector = %v, want [1 0 0]", got.Vector)
	}
	if got.Metadata["k"] != "v" {
		t.Errorf("Metadata[k] = %q, want %q", got.Metadata["k"], "v")
	}

	gotB, ok := idx2.Get("b")
	if !ok {
		t.Fatal("loaded index missing doc 'b'")
	}
	if gotB.Content != "world" {
		t.Errorf("Content = %q, want %q", gotB.Content, "world")
	}
}

func TestLoad_Merge(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(Document{ID: "existing", Content: "original", Vector: Vector{1}})

	jsonData := `{"documents":[{"id":"existing","content":"updated","content_hash":"","vector":[2]},{"id":"new","content":"fresh","content_hash":"","vector":[3]}]}`
	if err := idx.Load(strings.NewReader(jsonData)); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if idx.Len() != 2 {
		t.Fatalf("Len = %d, want 2 after merge", idx.Len())
	}
	got, _ := idx.Get("existing")
	if got.Content != "updated" {
		t.Errorf("existing doc Content = %q, want %q (should be upserted)", got.Content, "updated")
	}
	if _, ok := idx.Get("new"); !ok {
		t.Error("new doc should be present after merge")
	}
}

func TestAll(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(
		Document{ID: "a", Content: "one"},
		Document{ID: "b", Content: "two"},
		Document{ID: "c", Content: "three"},
	)
	all := idx.All()
	if len(all) != 3 {
		t.Fatalf("All() returned %d docs, want 3", len(all))
	}
	ids := make(map[string]bool)
	for _, d := range all {
		ids[d.ID] = true
	}
	for _, id := range []string{"a", "b", "c"} {
		if !ids[id] {
			t.Errorf("All() missing doc %q", id)
		}
	}
}

func TestConcurrentSafety(t *testing.T) {
	idx := NewIndex()
	_ = idx.Add(Document{ID: "seed", Content: "init", Vector: Vector{1, 0, 0}})

	var wg sync.WaitGroup
	const goroutines = 20
	const ops = 50

	// Writers: Add documents.
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			for j := 0; j < ops; j++ {
				id := string(rune('A'+n)) + string(rune('0'+j%10))
				_ = idx.Add(Document{ID: id, Content: "data", Vector: Vector{float32(n), float32(j), 0}})
			}
		}(i)
	}

	// Readers: Search concurrently.
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < ops; j++ {
				_ = idx.Search(Vector{1, 0, 0}, 5)
			}
		}()
	}

	// Removers: Remove documents concurrently.
	for i := 0; i < goroutines/2; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			for j := 0; j < ops; j++ {
				id := string(rune('A'+n)) + string(rune('0'+j%10))
				idx.Remove(id)
			}
		}(i)
	}

	// Readers: Len/Get/All concurrently.
	for i := 0; i < goroutines/2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < ops; j++ {
				_ = idx.Len()
				_, _ = idx.Get("seed")
				_ = idx.All()
			}
		}()
	}

	wg.Wait()
	// If we get here without a race condition panic, the test passes.
}
