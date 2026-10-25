package mer

import (
	"math/rand"
	"testing"
)

func TestWholeBatches(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	var exs []Example
	for p := range 200 {
		n := 1 + r.Intn(40) // some requests have more questions than a batch of their length holds
		for range n {
			exs = append(exs, Example{Parent: p, IDs: make([]int32, 1+r.Intn(5000))})
		}
	}
	seen := make([]bool, len(exs))
	batchOf := map[int]int{}
	nb := 0
	err := batches(exs, true, func(idx []int, B, T int) error {
		if len(idx) > B {
			t.Errorf("batch of %d rows holds %d examples", B, len(idx))
		}
		for _, i := range idx {
			if seen[i] {
				t.Errorf("example %d twice", i)
			}
			seen[i] = true
			if len(exs[i].IDs) > T {
				t.Errorf("example of %d tokens in a batch of %d", len(exs[i].IDs), T)
			}
			if b, ok := batchOf[exs[i].Parent]; ok && b != nb {
				t.Errorf("request %d in batches %d and %d", exs[i].Parent, b, nb)
			}
			batchOf[exs[i].Parent] = nb
		}
		nb++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, ok := range seen {
		if !ok {
			t.Fatalf("example %d left out", i)
		}
	}
}
