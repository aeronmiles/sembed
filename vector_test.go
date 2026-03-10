package sembed

import (
	"math"
	"testing"
)

const tolerance = 1e-6

func approxEqual(a, b float32) bool {
	return math.Abs(float64(a)-float64(b)) < tolerance
}

func TestCosineSimilarity_Identical(t *testing.T) {
	v := Vector{1, 2, 3}
	got := CosineSimilarity(v, v)
	if !approxEqual(got, 1.0) {
		t.Errorf("identical vectors: got %f, want 1.0", got)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := Vector{1, 0, 0}
	b := Vector{0, 1, 0}
	got := CosineSimilarity(a, b)
	if !approxEqual(got, 0.0) {
		t.Errorf("orthogonal vectors: got %f, want 0.0", got)
	}
}

func TestCosineSimilarity_Opposite(t *testing.T) {
	a := Vector{1, 2, 3}
	b := Vector{-1, -2, -3}
	got := CosineSimilarity(a, b)
	if !approxEqual(got, -1.0) {
		t.Errorf("opposite vectors: got %f, want -1.0", got)
	}
}

func TestCosineSimilarity_Empty(t *testing.T) {
	got := CosineSimilarity(Vector{}, Vector{})
	if got != 0.0 {
		t.Errorf("empty vectors: got %f, want 0.0", got)
	}
}

func TestCosineSimilarity_Nil(t *testing.T) {
	got := CosineSimilarity(nil, nil)
	if got != 0.0 {
		t.Errorf("nil vectors: got %f, want 0.0", got)
	}
}

func TestCosineSimilarity_DifferentLengths(t *testing.T) {
	a := Vector{1, 2, 3}
	b := Vector{1, 2}
	got := CosineSimilarity(a, b)
	if got != 0.0 {
		t.Errorf("different length vectors: got %f, want 0.0", got)
	}
}

func TestDot_KnownValues(t *testing.T) {
	tests := []struct {
		name string
		a, b Vector
		want float32
	}{
		{"simple", Vector{1, 2, 3}, Vector{4, 5, 6}, 32},
		{"unit", Vector{1, 0, 0}, Vector{0, 1, 0}, 0},
		{"negative", Vector{1, -1}, Vector{-1, 1}, -2},
		{"different lengths", Vector{1, 2}, Vector{1, 2, 3}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Dot(tt.a, tt.b)
			if !approxEqual(got, tt.want) {
				t.Errorf("Dot(%v, %v) = %f, want %f", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestNormalize_UnitVector(t *testing.T) {
	v := Vector{3, 4}
	norm := Normalize(v)
	mag := Magnitude(norm)
	if !approxEqual(mag, 1.0) {
		t.Errorf("normalized magnitude: got %f, want 1.0", mag)
	}
	// Check direction: 3/5, 4/5
	if !approxEqual(norm[0], 0.6) {
		t.Errorf("norm[0]: got %f, want 0.6", norm[0])
	}
	if !approxEqual(norm[1], 0.8) {
		t.Errorf("norm[1]: got %f, want 0.8", norm[1])
	}
}

func TestNormalize_Nil(t *testing.T) {
	got := Normalize(nil)
	if got != nil {
		t.Errorf("Normalize(nil) = %v, want nil", got)
	}
}

func TestNormalize_ZeroVector(t *testing.T) {
	got := Normalize(Vector{0, 0, 0})
	if got != nil {
		t.Errorf("Normalize(zero) = %v, want nil", got)
	}
}

func TestMagnitude(t *testing.T) {
	tests := []struct {
		name string
		v    Vector
		want float32
	}{
		{"3-4-5", Vector{3, 4}, 5},
		{"unit x", Vector{1, 0, 0}, 1},
		{"zero", Vector{0, 0}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Magnitude(tt.v)
			if !approxEqual(got, tt.want) {
				t.Errorf("Magnitude(%v) = %f, want %f", tt.v, got, tt.want)
			}
		})
	}
}
