package mer

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math/bits"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/cache"
)

// Bucket rounds a length up to one of a few sizes, so that few graphs are compiled.
func Bucket(n int) int {
	for _, b := range []int{64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192} {
		if n <= b {
			return b
		}
	}
	return (n + 1023) / 1024 * 1024
}

// BatchFor is how many sequences of T tokens go into one batch of the frozen backbone or of a prediction.
func BatchFor(T int) int { return max(1, min(32, 16384/T)) }

// byLength orders exs from the longest to the shortest.
func byLength(exs []Example) []int {
	order := make([]int, len(exs))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(exs[order[a]].IDs) > len(exs[order[b]].IDs) })
	return order
}

// batches cuts exs, from the longest, into batches of sequences of similar length: fn gets the examples of each, the
// bucketed length T and the batch size B. With whole, the questions of a request stay in one batch, which grows to
// hold them when they are more than BatchFor(T).
func batches(exs []Example, whole bool, fn func(idx []int, B, T int) error) error {
	if !whole {
		order := byLength(exs)
		for lo := 0; lo < len(order); {
			T := Bucket(len(exs[order[lo]].IDs))
			B := min(BatchFor(T), 1<<bits.Len(uint(len(order)-lo-1))) // the last batch is not padded to full size
			idx := order[lo:min(lo+B, len(order))]
			if err := fn(idx, B, T); err != nil {
				return err
			}
			lo += len(idx)
		}
		return nil
	}
	reqs := Requests(exs, byLength(exs))
	for lo := 0; lo < len(reqs); {
		T := Bucket(len(exs[reqs[lo][0]].IDs))
		B := BatchFor(T)
		idx := slices.Clone(reqs[lo])
		lo++
		for lo < len(reqs) && len(idx)+len(reqs[lo]) <= B {
			idx = append(idx, reqs[lo]...)
			lo++
		}
		if err := fn(idx, max(B, (len(idx)+7)/8*8), T); err != nil {
			return err
		}
	}
	return nil
}

// Requests gathers the examples order of exs by request, each request where its first example is in order.
func Requests(exs []Example, order []int) [][]int {
	at := map[int]int{}
	var out [][]int
	for _, i := range order {
		p := exs[i].Parent
		k, ok := at[p]
		if !ok {
			k = len(out)
			at[p] = k
			out = append(out, nil)
		}
		out[k] = append(out[k], i)
	}
	return out
}

// HiddenExec returns the backbone's final hidden states in bfloat16, as the cache keeps them.
func HiddenExec(backend compute.Backend, bb *backbone.Backbone) *model.Exec {
	exec := model.MustNewExec(backend, bb.Store, func(scope *model.Scope, tokens *Node) *Node {
		return ConvertDType(bb.HiddenStates(scope, tokens), dtypes.BFloat16)
	})
	exec.SetMaxCache(128)
	return exec
}

// HiddenStates computes the hidden states of exs with exec, the frozen bb, into a file in cacheDir named after name,
// these tokens and weights, or maps that file when it is already there; the file is read from the disk as batches need
// it. An empty cacheDir keeps the hidden states in memory, and an empty name prints nothing.
func HiddenStates(exec *model.Exec, bb *backbone.Backbone, exs []Example, cacheDir, name string) (*cache.VarCache, error) {
	lens := make([]int, len(exs))
	h := fnv.New64a()
	for i, e := range exs {
		lens[i] = len(e.IDs)
		_ = binary.Write(h, binary.LittleEndian, e.IDs) // writing to a hash never fails
	}
	path := ""
	if cacheDir != "" {
		weights := strings.ToLower(strings.ReplaceAll(bb.RepoID, "/", "_") + "-" + bb.DType.String())
		path = fmt.Sprintf("%s/mer-%s-%s-%x.bf16", cacheDir, weights, name, h.Sum64())
		if c, err := cache.OpenVarCache(path, lens, bb.HiddenDim); err == nil {
			fmt.Printf("  hidden states of %-6s loaded from cache\n", name)
			return c, nil
		}
	}
	var c *cache.VarCache
	if path == "" {
		c = cache.NewVarCache(lens, bb.HiddenDim)
	} else {
		var err error
		if c, err = cache.CreateVarCache(path, lens, bb.HiddenDim); err != nil {
			return nil, err
		}
	}
	t0, tokens := time.Now(), 0
	err := batches(exs, false, func(idx []int, B, T int) error {
		toks := padded(exs, idx, B, T, bb.PadID)
		for _, i := range idx {
			tokens += len(exs[i].IDs)
		}
		in := tensors.FromFlatDataAndDimensions(toks, B, T)
		defer func() { _ = in.FinalizeAll() }()
		out, err := exec.Call1(in)
		if err != nil {
			return err
		}
		defer func() { _ = out.FinalizeAll() }()
		return c.Put(idx, out)
	})
	if err != nil {
		_ = c.Close()
		return nil, err
	}
	if name != "" {
		el := time.Since(t0)
		fmt.Printf("  hidden states of %-6s: %d sequences, %d tokens in %v (%.0f tokens/s)\n", name, len(exs), tokens, el.Round(time.Second), float64(tokens)/el.Seconds())
	}
	if path != "" {
		if err := c.Commit(); err != nil {
			return nil, err
		}
	}
	return c, nil
}

// padded is the tokens of the examples idx of exs, each right-padded with pad to T tokens, in B rows.
func padded(exs []Example, idx []int, B, T int, pad int32) []int32 {
	toks := make([]int32, B*T)
	for i := range toks {
		toks[i] = pad
	}
	for r, i := range idx {
		copy(toks[r*T:], exs[i].IDs)
	}
	return toks
}
