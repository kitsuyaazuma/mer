// Latency measures how the time to answer one BFCL request grows with the number of tools, each question read in its
// own sequence after the whole state. The head is untrained because the time does not depend on the weights.
package main

import (
	"flag"
	"fmt"
	"log"
	"slices"
	"time"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/mer"
	"github.com/kitsuyaazuma/mer/internal/systemone"
	"github.com/kitsuyaazuma/mer/internal/toolbench"
)

const (
	bfclDir = "data/bfcl"
	calls   = 15 // timed calls per measurement
)

// toolCounts are the numbers of tools measured.
var toolCounts = []int{1, 2, 4, 8, 16, 32}

func must(err error) {
	if err != nil {
		log.Fatalf("%+v", err)
	}
}
func must1[T any](v T, err error) T { must(err); return v }

const usage = `Usage: go run ./cmd/latency

Measures the time Mer (routing_state, on Gemma 3 270M) takes to answer every question of the BFCL request
live_multiple_8-4-0, asked as the Index asks it, with 1 to 32 tools: one sequence per question, each with the whole
state. The head is untrained, because the time does not depend on its weights.
`

func main() {
	flag.Usage = func() { fmt.Fprint(flag.CommandLine.Output(), usage) }
	flag.Parse()
	backend := compute.MustNew()
	bb := must1(backbone.Load(backend, backbone.DefaultRepoID))
	d := must1(mer.ParseDesign("routing_state"))
	exec := d.LoRAExec(backend, bb)

	cases := must1(toolbench.LoadBFCL(bfclDir, "live_multiple"))
	start := slices.IndexFunc(cases, func(c toolbench.Case) bool { return c.ID == "live_multiple_8-4-0" })
	base := cases[start]
	var pool []toolbench.Tool
	seen := map[string]bool{}
	for _, c := range cases[start:] {
		for _, t := range c.Tools {
			id := toolbench.ToolQuestion(t.Name, false, false).ID // todo_add and todo.add ask the same question
			if !seen[id] {
				seen[id] = true
				pool = append(pool, t)
			}
		}
	}

	fmt.Printf("request %s: %q, median of %d calls\n", base.ID, base.Query, calls)
	fmt.Printf("%6s | %10s | %9s | %7s\n", "tools", "time (ms)", "sequences", "tokens")
	for _, n := range toolCounts {
		c := base
		c.Tools = pool[:n]
		r := toolbench.FromBFCL(c)
		exs := must1(mer.NewExamples(bb, []systemone.Request{r}, true))
		tokens := 0
		for _, e := range exs {
			tokens += len(e.IDs)
		}
		answer := func() {
			must1(mer.Predict(exec, exs, false, func(idx []int, B, T, K int) ([]*tensors.Tensor, error) {
				return d.LoRAInputs(exs, bb.PadID, idx, B, T, K), nil
			}))
		}
		for range 3 {
			answer()
		}
		ts := make([]time.Duration, calls)
		for i := range ts {
			t0 := time.Now()
			answer()
			ts[i] = time.Since(t0)
		}
		slices.Sort(ts)
		fmt.Printf("%6d | %10.1f | %9d | %7d\n", n, float64(ts[len(ts)/2].Microseconds())/1000, len(exs), tokens)
	}
}
