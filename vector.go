package sembed

import "math"

// CosineSimilarity returns the cosine similarity between two vectors.
// Returns 0 if either vector is zero-length, different lengths, or zero magnitude.
func CosineSimilarity(a, b Vector) float32 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0
	}
	dot := Dot(a, b)
	magA := Magnitude(a)
	magB := Magnitude(b)
	if magA == 0 || magB == 0 {
		return 0
	}
	return dot / (magA * magB)
}

// Dot returns the dot product of two vectors.
// Returns 0 if vectors have different lengths.
func Dot(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Normalize returns a unit vector in the same direction.
// Returns nil if the input is nil or zero magnitude.
func Normalize(v Vector) Vector {
	if v == nil {
		return nil
	}
	mag := Magnitude(v)
	if mag == 0 {
		return nil
	}
	result := make(Vector, len(v))
	for i := range v {
		result[i] = v[i] / mag
	}
	return result
}

// Magnitude returns the L2 norm of the vector.
func Magnitude(v Vector) float32 {
	var sum float64
	for _, val := range v {
		sum += float64(val) * float64(val)
	}
	return float32(math.Sqrt(sum))
}
