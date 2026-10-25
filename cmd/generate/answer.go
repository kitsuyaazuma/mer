package main

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/massive"
)

func runAnswer(backend compute.Backend, bb *backbone.Backbone, utts []massive.Utterance) {
	const bs, tokens = 32, 4
	sch := massive.Scenarios10()
	sch.Answer = "\nANSWER:"
	test := must1(massive.BuildSplitsWith(bb.Tok, sch, utts, massive.Scenario, massive.SplitOptions{Seed: seed})).Test
	prompts := make([][]int32, len(test))
	for i, p := range test {
		prompts[i] = p.IDs
	}
	gens, _ := generate(nextTokenExec(backend, bb), bb.Tok, prompts, bs, tokens, nil)

	var ok, unparsed int
	counts := map[string]int{}
	for i, p := range test {
		gen := strings.TrimSpace(gens[i])
		counts[gen]++
		id := parseID(sch, gen)
		switch id {
		case "":
			unparsed++
		case sch.Options[p.Canon].ID:
			ok++
		}
	}
	n := len(test)
	fmt.Printf("frozen %s, generate %d tokens greedily after \"ANSWER:\", no training | test=%d\n", bb.HF.Repo.ID, tokens, n)
	fmt.Printf("accuracy %.4f | not an option ID: %d (%.1f%%) | chance %.3f\n", float64(ok)/float64(n), unparsed, 100*float64(unparsed)/float64(n), 1/float64(len(sch.Options)))
	printTop(counts, 8)
}

// parseID returns the longest option ID that gen starts with, or "".
func parseID(s massive.Schema, gen string) string {
	best := ""
	for _, o := range s.Options {
		if strings.HasPrefix(gen, o.ID) && len(o.ID) > len(best) {
			best = o.ID
		}
	}
	return best
}
