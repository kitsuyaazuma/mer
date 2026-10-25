// Prepare writes BFCL, ToolACE and When2Call as the labeled System One datasets cmd/mer trains and tests on, in the
// form of the Jev Decision Index.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"

	"github.com/kitsuyaazuma/mer/internal/systemone"
	"github.com/kitsuyaazuma/mer/internal/toolbench"
)

var flagOut = flag.String("out", "data/mer", "directory to write the datasets into")

const usage = `Usage: go run ./cmd/prepare [flags]

Writes the datasets of the article as JSONL, one labeled System One request per line, as the Jev Decision Index asks
them: for a BFCL case, the conversation and the published function schemas as the state and a yes / no choice per
tool; for a When2Call item, the tools and the question as the state and a choice among the responses.
  train.jsonl  8,000 requests: ToolACE calls (50%), When2Call refusals (20%) asked like BFCL, and When2Call
               preference pairs (30%) asked like When2Call
  dev.jsonl    600 more of the same mixture
  bfcl.jsonl   the 1,694 BFCL cases of the Index (simple, live_simple, live_multiple)
  w2c.jsonl    the 3,652 When2Call test items of the Index
  bval.jsonl   BFCL multiple, outside the Index, where cmd/mer chooses a threshold on P(yes)

Training requests show at most 8 tools (the invoked ones kept), and those with a tool named like a BFCL tool are left
out.

Flags:
`

const (
	dataDir  = "data" // bfcl/, toolace/ and when2call/
	trainN   = 8000
	devN     = 600
	maxTools = 8
	seed     = 1
)

// shares are the parts of ToolACE, When2Call pref and When2Call sft in train and dev.
var shares = []float64{0.5, 0.3, 0.2}

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
	must(os.MkdirAll(*flagOut, 0o755))

	bfclNames := map[string]bool{}
	var bfcl, bval []systemone.Request
	for _, cat := range slices.Concat(toolbench.CallCategories, toolbench.IrrelevantCategories) {
		for _, c := range must1(toolbench.LoadBFCL(dataDir+"/bfcl", cat)) {
			for _, t := range c.Tools {
				bfclNames[toolbench.NormName(t.Name)] = true
			}
			if slices.Contains(toolbench.IndexCategories, cat) && c.GoldInTools() {
				bfcl = append(bfcl, toolbench.FromBFCL(c))
			}
			if cat == "multiple" {
				bval = append(bval, toolbench.FromBFCL(c))
			}
		}
	}
	var w2c []systemone.Request
	for _, c := range must1(toolbench.LoadWhen2CallTest(dataDir + "/when2call/test/when2call_test_mcq.jsonl")) {
		w2c = append(w2c, toolbench.FromWhen2CallTest(c))
	}

	pools, dropped := trainingPools(bfclNames, rand.New(rand.NewSource(seed)))
	fmt.Printf("training pools: toolace %d, when2call pref %d, sft %d (left out, a tool named like a BFCL tool: %v)\n", len(pools[0]), len(pools[1]), len(pools[2]), dropped)
	train, dev := take(&pools, trainN), take(&pools, devN)

	for _, s := range []struct {
		name string
		rs   []systemone.Request
	}{{"train", train}, {"dev", dev}, {"bfcl", bfcl}, {"w2c", w2c}, {"bval", bval}} {
		path := filepath.Join(*flagOut, s.name+".jsonl")
		must(systemone.WriteFile(path, s.rs))
		must1(systemone.ReadFile(path)) // every request must read back as a valid one
		fmt.Printf("%-24s %5d requests\n", path, len(s.rs))
	}
}

// trainingPools are the ToolACE, When2Call pref and When2Call sft requests, each shuffled, without those that show a
// tool named like a BFCL tool.
func trainingPools(bfclNames map[string]bool, rng *rand.Rand) (pools [3][]systemone.Request, dropped map[string]int) {
	dropped = map[string]int{}
	add := func(k int, src string, i int, tools []toolbench.Tool, build func(id string) (systemone.Request, bool)) {
		for _, t := range tools {
			if bfclNames[toolbench.NormName(t.Name)] {
				dropped[src]++
				return
			}
		}
		if r, ok := build(fmt.Sprintf("%s-%d", src, i)); ok {
			pools[k] = append(pools[k], r)
		}
	}
	for i, t := range must1(toolbench.LoadToolACE(dataDir + "/toolace/data.json")) {
		add(0, "toolace", i, t.Tools, func(id string) (systemone.Request, bool) { return toolbench.FromToolACE(id, t, maxTools, rng), true })
	}
	for i, t := range must1(toolbench.LoadWhen2CallPref(dataDir + "/when2call/train/when2call_train_pref.jsonl")) {
		add(1, "pref", i, t.Tools, func(id string) (systemone.Request, bool) {
			return toolbench.FromWhen2CallPref(id, t, maxTools, rng), true
		})
	}
	for i, t := range must1(toolbench.LoadWhen2CallTrain(dataDir + "/when2call/train/when2call_train_sft.jsonl")) {
		add(2, "sft", i, t.Tools, func(id string) (systemone.Request, bool) { return toolbench.FromWhen2CallSFT(id, t, maxTools, rng) })
	}
	for _, xs := range pools {
		rng.Shuffle(len(xs), func(i, j int) { xs[i], xs[j] = xs[j], xs[i] })
	}
	return pools, dropped
}

// take removes n requests from the front of the pools, in the mixture of shares.
func take(pools *[3][]systemone.Request, n int) []systemone.Request {
	var out []systemone.Request
	for k := range pools {
		want := min(int(math.Round(shares[k]*float64(n))), len(pools[k]))
		out = append(out, pools[k][:want]...)
		pools[k] = pools[k][want:]
	}
	return out
}
