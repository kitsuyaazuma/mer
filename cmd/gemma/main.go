// Gemma loads Gemma 3 270M with GoMLX, prints the tokens and the hidden states of a Japanese sentence,
// and generates text greedily to check that the model still works as a language model.
package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"

	"github.com/kitsuyaazuma/mer/internal/backbone"
)

const usage = `Usage: go run ./cmd/gemma

Loads Gemma 3 270M with internal/backbone, prints the tokens of a Japanese sentence and the shape of its hidden states, and generates 20 tokens greedily through the tied embeddings to show that the model still works as a language model.
`

func main() {
	flag.Usage = func() { fmt.Fprint(flag.CommandLine.Output(), usage) }
	flag.Parse()
	backend := compute.MustNew()
	fmt.Println("backend:", backend.Description())

	t0 := time.Now()
	bb, err := backbone.Load(backend, backbone.DefaultRepoID)
	if err != nil {
		log.Fatalf("%+v", err)
	}
	fmt.Printf("%s | loaded in %v\n", bb.Describe(), time.Since(t0).Round(time.Millisecond))

	text := "明日の東京の天気を教えて"
	ids := bb.Tok.Encode(text)
	fmt.Printf("\n%q -> %v\n", text, ids)
	for _, id := range ids {
		fmt.Printf("  %6d %q\n", id, bb.Tok.Decode([]int{id}))
	}

	hiddenExec := model.MustNewExec(backend, bb.Store, func(scope *model.Scope, tokens *Node) *Node {
		return bb.HiddenStates(scope, tokens)
	})
	hidden := hiddenExec.MustCall1(tensors.FromValue([][]int32{toInt32(ids)}))
	fmt.Printf("\nhidden states shape: %s\n", hidden.Shape())

	// Gemma ties its LM head to the embedding table, so the logits are hidden @ E^T.
	emb := bb.Store.GetVariable("/token_embed/embeddings")
	nextToken := model.MustNewExec(backend, bb.Store, func(scope *model.Scope, tokens *Node) *Node {
		h := bb.HiddenStates(scope, tokens)
		logits := MatMul(h, Transpose(emb.NodeValue(tokens.Graph()), 0, 1)) // [1, T, 640] x [640, vocab]
		return ArgMax(logits, -1, dtypes.Int32)
	})
	gen := ids        // continue the same sentence
	const bufLen = 48 // fixed shape (one compilation); right padding is harmless with a causal mask
	for range 20 {
		buf := make([]int, bufLen)
		copy(buf, gen)
		pred := tensors.MustCopyFlatData[int32](nextToken.MustCall1(tensors.FromValue([][]int32{toInt32(buf)})))
		gen = append(gen, int(pred[len(gen)-1]))
	}
	fmt.Printf("\ngreedy generation: %q\n", bb.Tok.Decode(gen))
}

func toInt32(ids []int) []int32 {
	out := make([]int32, len(ids))
	for i, id := range ids {
		out[i] = int32(id)
	}
	return out
}
