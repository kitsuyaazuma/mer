package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/massive"
)

func toInt32(ids []int) []int32 {
	out := make([]int32, len(ids))
	for i, id := range ids {
		out[i] = int32(id)
	}
	return out
}

func runSchema(backend compute.Backend, bb *backbone.Backbone, utts []massive.Utterance) {
	const bs, tokens = 16, 8
	sch := massive.Scenarios10()
	inSchema := map[string]bool{}
	var ids []string
	for _, o := range sch.Options {
		inSchema[o.ID] = true
		ids = append(ids, fmt.Sprintf("%q", o.ID))
	}

	var sb strings.Builder
	sb.WriteString("ユーザーの発話を、次の機能のどれか1つに振り分けて、JSON で答えてください。\n\n機能:\n")
	for _, o := range sch.Options {
		fmt.Fprintf(&sb, "- %s: %s\n", o.ID, o.Desc)
	}
	sb.WriteString("\n出力は次の JSON Schema に従ってください。\n")
	fmt.Fprintf(&sb, `{"type": "object", "properties": {"intent": {"type": "string", "enum": [%s]}}, "required": ["intent"]}`+"\n\n", strings.Join(ids, ", "))
	if *flagShots > 0 {
		rng := rand.New(rand.NewSource(seed))
		byLabel := map[string][]string{}
		for _, u := range utts {
			if u.Partition == "train" && inSchema[u.Scenario] {
				byLabel[u.Scenario] = append(byLabel[u.Scenario], u.Text)
			}
		}
		order := rng.Perm(len(sch.Options))
		for _, k := range order[:min(*flagShots, len(order))] {
			id := sch.Options[k].ID
			texts := byLabel[id]
			ex := texts[rng.Intn(len(texts))]
			fmt.Fprintf(&sb, "発話: %s\n", ex)
			fmt.Fprintf(&sb, `出力: {"intent": "%s"}`+"\n\n", id)
		}
	}
	header := sb.String()
	if *flagShow {
		fmt.Print("---- prompt header ----\n" + header + "---- end ----\n")
	}

	var prompts, cls [][]int32
	var golds []string
	for _, u := range utts {
		if u.Partition == "test" && inSchema[u.Scenario] {
			prompts = append(prompts, toInt32(bb.Tok.Encode(header+"発話: "+u.Text+"\n"+`出力: {"intent": "`)))
			cls = append(cls, toInt32(bb.Tok.Encode("STATE:\n"+u.Text)))
			golds = append(golds, u.Scenario)
		}
	}

	stopAtQuote := func(gen string) bool { return strings.Contains(gen, `"`) }
	gens, genTime := generate(nextTokenExec(backend, bb), bb.Tok, prompts, bs, tokens, stopAtQuote)
	var ok, unparsed, promptTokens, clsTokens int
	counts := map[string]int{}
	for i, gen := range gens {
		promptTokens += len(prompts[i])
		clsTokens += len(cls[i])
		counts[gen]++
		ans, _, closed := strings.Cut(gen, `"`)
		switch {
		case !closed || !inSchema[ans]:
			unparsed++
		case ans == golds[i]:
			ok++
		}
	}

	// The classifier's forward alone: its Linear head (cmd/intent) is negligible.
	forward := model.MustNewExec(backend, bb.Store, func(scope *model.Scope, tokens *Node) *Node {
		return ReduceSum(bb.HiddenStates(scope, tokens), -1) // the hidden states, summed only to make the output small
	})
	forwardAll := func() (elapsed time.Duration) {
		for lo := 0; lo < len(cls); lo += bs {
			batch := cls[lo:min(lo+bs, len(cls))]
			toks, T := pad(batch, 0)
			t0 := time.Now()
			in := tensors.FromFlatDataAndDimensions(toks, len(batch), T)
			_ = forward.MustCall1(in).FinalizeAll()
			elapsed += time.Since(t0)
		}
		return elapsed
	}
	forwardAll() // the first pass compiles the shapes
	clsTime := forwardAll()

	n := len(prompts)
	fmt.Printf("frozen %s, JSON Schema prompt, %d solved examples, answer prefilled up to {\"intent\": \", greedy until the closing quote (at most %d tokens) | test=%d\n", bb.HF.Repo.ID, *flagShots, tokens, n)
	fmt.Printf("accuracy %.4f (%d) | not one of the 10 IDs: %d (%.1f%%) | chance 0.100\n", float64(ok)/float64(n), ok, unparsed, 100*float64(unparsed)/float64(n))
	printTop(counts, 10)
	fmt.Printf("prompt tokens per utterance %.0f | generation: %.1f ms per utterance (bs %d, no KV cache)\n",
		float64(promptTokens)/float64(n), genTime.Seconds()*1000/float64(n), bs)
	fmt.Printf("classifier input tokens per utterance %.0f | one forward: %.1f ms per utterance (bs %d)\n",
		float64(clsTokens)/float64(n), clsTime.Seconds()*1000/float64(n), bs)
}
