// Generate makes the frozen Gemma 3 270M generate the function ID of each MASSIVE ja-JP test utterance of the 10 functions, and scores it.
// It has no KV cache, so its generation times are an upper bound.
package main

import (
	"flag"
	"fmt"
	"log"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-huggingface/tokenizers"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/massive"
)

var (
	flagPrompt = flag.String("prompt", "schema", "answer (an empty answer slot) or schema (a JSON Schema of the answer)")
	flagShots  = flag.Int("shots", 0, "with -prompt schema: solved examples from train put in the prompt, one per function (0 or 10)")
	flagShow   = flag.Bool("show", false, "with -prompt schema: print the shared part of the prompt before the results")
)

const usage = `Usage: go run ./cmd/generate [flags]

Makes the frozen Gemma 3 270M generate the function ID of each MASSIVE ja-JP test utterance of the 10 functions, and scores it.

Examples (the commands of the article):
  # the functions and an empty answer slot
  go run ./cmd/generate -prompt answer
  # the functions and a JSON Schema of the answer
  go run ./cmd/generate -prompt schema -shots 0
  # the same with one solved example per function
  go run ./cmd/generate -prompt schema -shots 10

Flags:
`

// seed fixes the option orders (as in cmd/intent) and the choice of the solved examples.
const seed = 1

func must(err error) {
	if err != nil {
		log.Fatalf("%+v", err)
	}
}
func must1[T any](v T, err error) T { must(err); return v }

func main() {
	flag.Usage = func() {
		fmt.Fprint(flag.CommandLine.Output(), usage)
		flag.PrintDefaults()
	}
	flag.Parse()
	backend := compute.MustNew()
	bb := must1(backbone.Load(backend, backbone.DefaultRepoID))
	path := must1(massive.EnsureJA("data/massive"))
	utts := must1(massive.Load(path))
	switch *flagPrompt {
	case "answer":
		runAnswer(backend, bb, utts)
	case "schema":
		runSchema(backend, bb, utts)
	default:
		log.Fatalf("unknown -prompt %q (answer | schema)", *flagPrompt)
	}
}

func nextTokenExec(backend compute.Backend, bb *backbone.Backbone) *model.Exec {
	return model.MustNewExec(backend, bb.Store, func(scope *model.Scope, tokens, last *Node) *Node {
		g := tokens.Graph()
		h := bb.HiddenStates(scope, tokens) // [B, T, hidden]
		B, T := tokens.Shape().Dimensions[0], tokens.Shape().Dimensions[1]
		pos := Iota(g, shapes.Make(dtypes.Int32, B, T), 1)                    // pos[b][t] = t
		sel := ConvertDType(Equal(pos, ExpandAxes(last, -1)), dtypes.Float32) // 1 only where t == last[b]
		hLast := Einsum("bt,bth->bh", sel, h)                                 // h[b, last[b], :]
		emb := bb.Store.GetVariable("/token_embed/embeddings").NodeValue(g)
		logits := Einsum("bh,vh->bv", hLast, emb) // the tied LM head: [B, vocab]
		return ArgMax(logits, -1, dtypes.Int32)
	})
}

// generate greedily adds up to n tokens to each prompt, bs prompts at a time, and returns the generated texts.
// A prompt stops early once stop (if not nil) accepts its text. elapsed counts only the Gemma calls.
func generate(next *model.Exec, tok tokenizers.Tokenizer, prompts [][]int32, bs, n int, stop func(string) bool) (texts []string, elapsed time.Duration) {
	texts = make([]string, len(prompts))
	for lo := 0; lo < len(prompts); lo += bs {
		batch := prompts[lo:min(lo+bs, len(prompts))]
		toks, T := pad(batch, n)
		last := make([]int32, len(batch))
		for b, p := range batch {
			last[b] = int32(len(p) - 1)
		}
		gen := make([][]int, len(batch))
		done := make([]bool, len(batch))
		t0 := time.Now()
		for range n {
			in := tensors.FromFlatDataAndDimensions(toks, len(batch), T)
			out := tensors.MustCopyFlatData[int32](next.MustCall1(in, tensors.FromValue(last)))
			for b := range batch {
				if done[b] {
					continue
				}
				gen[b] = append(gen[b], int(out[b]))
				last[b]++
				toks[b*T+int(last[b])] = out[b]
				if stop != nil && stop(tok.Decode(gen[b])) {
					done[b] = true
				}
			}
			if !slices.Contains(done, false) {
				break
			}
		}
		elapsed += time.Since(t0)
		for b := range batch {
			texts[lo+b] = tok.Decode(gen[b])
		}
	}
	return texts, elapsed
}

// pad right-pads the prompts into one [len(batch), T] batch, T being maxLen+extra rounded up to a multiple of 32.
func pad(batch [][]int32, extra int) (toks []int32, T int) {
	maxLen := 0
	for _, p := range batch {
		maxLen = max(maxLen, len(p))
	}
	T = (maxLen + extra + 31) / 32 * 32
	toks = make([]int32, len(batch)*T)
	for b, p := range batch {
		copy(toks[b*T:], p)
	}
	return toks, T
}

func printTop(counts map[string]int, n int) {
	keys := slices.Collect(maps.Keys(counts))
	slices.SortFunc(keys, func(a, b string) int {
		if counts[a] != counts[b] {
			return counts[b] - counts[a] // the more frequent first
		}
		return strings.Compare(a, b) // equal counts: alphabetical, as the map order changes from run to run
	})
	fmt.Print("most frequent generations:")
	for _, k := range keys[:min(n, len(keys))] {
		fmt.Printf(" %q x%d", k, counts[k])
	}
	fmt.Println()
}
