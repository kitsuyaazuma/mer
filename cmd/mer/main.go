// Mer trains and tests a Clef-style Decision Model, a backbone (frozen, or with LoRA adapters) and a head that answer
// the questions of labeled System One requests, and can save the trained model for cmd/serve.
package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/ml/model"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/cache"
	"github.com/kitsuyaazuma/mer/internal/mer"
	"github.com/kitsuyaazuma/mer/internal/systemone"
)

var (
	flagData     = flag.String("data", "data/mer", "directory of the datasets: train.jsonl, dev.jsonl and those of -test and -calib")
	flagTest     = flag.String("test", "bfcl,w2c", "comma separated datasets to test on")
	flagCalib    = flag.String("calib", "bval", "dataset where a threshold on P(yes) of the yes / no questions is chosen, for case@tau (0.5 if empty)")
	flagDesigns  = flag.String("designs", "linear,routing_state", "comma separated designs (see above)")
	flagRuns     = flag.Int("runs", 3, "training runs per design, each with its own seed (with -lora: always 1)")
	flagCacheDir = flag.String("cache_dir", "", "directory where the hidden states and the tokenized requests are kept between runs; the hidden states are read from there as training needs them (in memory if empty)")
	flagDump     = flag.String("dump", "", "directory for the probabilities of run 0 on the -test datasets (none if empty)")
	flagLoRA     = flag.Bool("lora", false, "also train LoRA adapters of the backbone, together with the head (one design, one run; no hidden-state cache)")
	flagLoRARun  = flag.Int("lora_run", 0, "with -lora: which run this is; the seed of the weights and of dropout is 1000 + lora_run")
	flagBackbone = flag.String("backbone", backbone.Default, "the backbone: gemma3-270m, gemma4-e4b or a Hugging Face repository ID")
	flagDType    = flag.String("dtype", "float32", "dtype of the backbone weights: float32, or bfloat16 to halve the memory on a GPU")
	flagMaxLen   = flag.Int("max_len", 0, "leave out the training and dev sequences longer than this many tokens, to bound the memory of a batch (no limit if 0)")
	flagSave     = flag.String("save", "", "with one design: an empty directory to save the model of run 0 into, for cmd/serve (none if empty)")
	flagLoRALR   = flag.Float64("lora_lr", 2e-4, "with -lora: the learning rate (the peak one with -warmup or -cosine)")
	flagWarmup   = flag.Int("warmup", 0, "with -lora: steps over which the learning rate rises from 0 to -lora_lr")
	flagCosine   = flag.Bool("cosine", false, "with -lora: after the warmup, lower the learning rate along a cosine to 0 by the last step")
	flagClip     = flag.Float64("clip", 0, "with -lora: clip every value of each Adam step to plus or minus this (no clipping if 0)")
	flagMaxSteps = flag.Int("max_steps", 0, "with -lora: stop training after this many steps, to compare settings quickly (the whole epoch if 0)")
)

const usage = `Usage: go run ./cmd/mer [flags]

Trains Mer, a Decision Model made of a backbone and a head, on the labeled System One requests of -data/train.jsonl
(written by cmd/prepare, or any dataset of systemone.Record lines), and tests it on dev.jsonl and the -test datasets. Every
question is read in its own sequence, after the whole state.

Designs (-designs) are the heads: linear, routing (Evidence Routing over every token) or routing_state (over the
state only). With _cross after the name (routing_state_cross), the questions of a request also look at each other
before they are scored (Cross-field); they are then batched together, and with -lora a training batch holds a whole
request even when it is longer than the token budget.

Examples (the commands of the article):
  go run ./cmd/prepare
  # the frozen backbone
  go run ./cmd/mer -cache_dir ./cache -dump results/mer_dump -designs linear,routing_state
  # LoRA adapters trained with the head, saved for cmd/serve
  go run ./cmd/mer -lora -cache_dir ./cache -designs routing_state -save models/routing_state
  # the Cross-field
  go run ./cmd/mer -cache_dir ./cache -designs linear_cross,routing_state_cross
  go run ./cmd/mer -lora -cache_dir ./cache -designs routing_state_cross

Flags:
`

// The settings of the article's runs.
const (
	batchSize  = 16
	epochs     = 10   // of the head on the frozen backbone
	loraBudget = 4096 // with -lora: tokens per training batch
	seed       = 1    // the shuffles of training
)

func must(err error) {
	if err != nil {
		log.Fatalf("%+v", err)
	}
}
func must1[T any](v T, err error) T { must(err); return v }

// split is one dataset, its requests and their sequences, with the frozen backbone's hidden states unless -lora.
type split struct {
	name  string
	rs    []systemone.Request
	exs   []mer.Example
	cache *cache.VarCache
}

func main() {
	flag.Usage = func() {
		fmt.Fprint(flag.CommandLine.Output(), usage)
		flag.PrintDefaults()
	}
	flag.Parse()
	t0 := time.Now()
	var designs []mer.Design
	for _, name := range strings.Split(*flagDesigns, ",") {
		designs = append(designs, must1(mer.ParseDesign(strings.TrimSpace(name))))
	}
	if (*flagLoRA || *flagSave != "") && len(designs) != 1 {
		log.Fatalf("-lora and -save take one design")
	}
	backend := compute.MustNew()
	bb := must1(backbone.LoadDType(backend, must1(backbone.RepoID(*flagBackbone)), must1(backbone.ParseDType(*flagDType))))
	fmt.Println(bb.Describe())
	if *flagCacheDir != "" {
		mer.LoadTokenCache(*flagCacheDir)
	}
	if *flagLoRA {
		*flagRuns = 1
		mer.SetLoRA(bb)
	}

	// the datasets: train, dev, the tests, then calib
	names := []string{"train", "dev"}
	for _, name := range strings.Split(*flagTest, ",") {
		if name = strings.TrimSpace(name); name != "" {
			names = append(names, name)
		}
	}
	if *flagCalib != "" {
		names = append(names, *flagCalib)
	}
	hidden := mer.HiddenExec(backend, bb)
	var splits []*split
	for _, name := range names {
		splits = append(splits, load(bb, hidden, name))
	}
	reported := names[1:] // the scores of calib are not reported
	if *flagCalib != "" {
		reported = names[1 : len(names)-1]
	}
	t := &trainer{backend: backend, bb: bb}

	results := map[string][]map[string]scores{}
	for _, d := range designs {
		for r := range *flagRuns {
			run := r
			if *flagLoRA {
				run = *flagLoRARun
			}
			t1 := time.Now()
			var predict predictor
			var params int
			if *flagLoRA {
				predict, params = t.trainLoRA(d, run, splits[0])
			} else {
				predict, params = t.train(d, run, splits[0])
			}
			res := test(d, run, splits, predict)
			results[d.Name] = append(results[d.Name], res)
			for _, name := range reported {
				fmt.Printf("RESULT design=%s run=%d split=%s | %s\n", d.Name, run, name, res[name])
			}
			fmt.Printf("  %s run %d: %d trained parameters, %.0f s\n", d.Name, run, params, time.Since(t1).Seconds())
		}
	}
	summary(designs, reported, results)
	fmt.Printf("total wall time %v\n", time.Since(t0).Round(time.Second))
}

// load reads one dataset and writes its questions as sequences, with the frozen backbone's hidden states unless -lora.
// The unlabeled questions of the tests only go to -dump; training and dev sequences longer than -max_len are left out.
func load(bb *backbone.Backbone, hidden *model.Exec, name string) *split {
	s := &split{name: name, rs: must1(systemone.ReadFile(filepath.Join(*flagData, name+".jsonl")))}
	keep := *flagDump != "" && name != "train" && name != "dev"
	s.exs = must1(mer.NewExamples(bb, s.rs, keep))
	must(mer.SaveTokenCache())
	if *flagMaxLen > 0 && (name == "train" || name == "dev") {
		n := len(s.exs)
		s.exs = slices.DeleteFunc(s.exs, func(e mer.Example) bool { return len(e.IDs) > *flagMaxLen })
		if n > len(s.exs) {
			fmt.Printf("%s: %d sequences left out, longer than %d tokens\n", name, n-len(s.exs), *flagMaxLen)
		}
	}
	describe(s)
	if name == "train" {
		e := s.exs[0]
		fmt.Printf("\n---- a rendered question (%s %s) ----\n%s\n----\n", e.Req.ID, e.Question().ID, e.Req.Render(e.Q).Text)
	}
	if !*flagLoRA {
		s.cache = must1(mer.HiddenStates(hidden, bb, s.exs, *flagCacheDir, name))
	}
	return s
}

// test predicts every dataset but train once and scores it, with the threshold on P(yes) chosen on calib.
func test(d mer.Design, run int, splits []*split, predict predictor) map[string]scores {
	pq := make([][][][]float32, len(splits))
	lens := make([][]int, len(splits))
	for k := 1; k < len(splits); k++ {
		pq[k] = mer.PerQuestion(splits[k].rs, splits[k].exs, predict(splits[k]))
		lens[k] = make([]int, len(splits[k].rs))
		for _, e := range splits[k].exs {
			lens[k][e.Parent] = max(lens[k][e.Parent], len(e.IDs))
		}
	}
	tau := 0.5
	if *flagCalib != "" {
		calib := len(splits) - 1
		best := -1.0
		for t := 1; t < 100; t++ {
			if a := evaluate(splits[calib].rs, pq[calib], lens[calib], float64(t)/100).tauAcc(); a > best {
				best, tau = a, float64(t)/100
			}
		}
	}
	out := map[string]scores{}
	for k := 1; k < len(splits); k++ {
		s := splits[k]
		out[s.name] = evaluate(s.rs, pq[k], lens[k], tau)
		if *flagDump != "" && run == 0 && s.name != "dev" && s.name != *flagCalib {
			dump(filepath.Join(*flagDump, d.Name+"-"+s.name+".csv"), s.rs, pq[k])
		}
	}
	return out
}

func describe(s *split) {
	src := map[string]int{}
	for _, r := range s.rs {
		src[strings.SplitN(r.Source, ":", 2)[0]]++
	}
	var lens []int
	labeled := 0
	for _, e := range s.exs {
		lens = append(lens, len(e.IDs))
		if e.Question().Gold >= 0 {
			labeled++
		}
	}
	slices.Sort(lens)
	pct := func(p float64) int { return lens[min(len(lens)-1, int(p*float64(len(lens))))] }
	fmt.Printf("%-5s %5d requests %v | %6d sequences (%d labeled) | tokens med %d p95 %d max %d\n",
		s.name, len(s.rs), src, len(s.exs), labeled, pct(0.5), pct(0.95), lens[len(lens)-1])
}
