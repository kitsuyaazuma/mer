package cache

import (
	"path/filepath"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
)

func TestFileRoundTrip(t *testing.T) {
	lens, H := []int{2, 3}, 4
	path := filepath.Join(t.TempDir(), "c.bf16")
	c, err := CreateVarCache(path, lens, H)
	if err != nil {
		t.Fatal(err)
	}
	// a batch of 2 rows of 3 tokens; each value byte is its position, so every copy is checked
	hidden := tensors.FromShape(shapes.Make(dtypes.BFloat16, 2, 3, H))
	if err := hidden.MutableBytes(func(b []byte) {
		for i := range b {
			b[i] = byte(i)
		}
	}); err != nil {
		t.Fatal(err)
	}
	if err := c.Put([]int{0, 1}, hidden); err != nil {
		t.Fatal(err)
	}
	if err := c.Commit(); err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := OpenVarCache(path, []int{2, 4}, H); err == nil {
		t.Error("lengths that differ must not open")
	}
	got, err := c.Gather([]int{1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	row := H * 2
	if err := got.ConstBytes(func(b []byte) {
		for k := range 3 * row { // example 1: all its 3 tokens, from row 1 of the batch
			if b[k] != byte(3*row+k) {
				t.Fatalf("example 1 byte %d = %d", k, b[k])
			}
		}
		for k := range 2 * row { // example 0: its 2 tokens, then zeros
			if b[3*row+k] != byte(k) {
				t.Fatalf("example 0 byte %d = %d", k, b[3*row+k])
			}
		}
		for k := 5 * row; k < 6*row; k++ {
			if b[k] != 0 {
				t.Fatalf("padding byte %d = %d", k, b[k])
			}
		}
	}); err != nil {
		t.Fatal(err)
	}
}
