// Intent assigns MASSIVE ja-JP utterances to functions with a linear layer (fixed) or by the cosine with their descriptions (cosine).
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/embgemma"
	"github.com/kitsuyaazuma/mer/internal/massive"
)

var (
	flagLabels   = flag.String("labels", "scenario10", "scenario10 (10 functions) or intent60 (all 60 MASSIVE intents)")
	flagMethods  = flag.String("methods", "fixed,cosine", "comma separated: fixed (a trained linear layer) and cosine (the cosine with the descriptions, no training)")
	flagCands    = flag.Int("candidates", 0, "if >0, show only this many options per utterance: the correct one and random wrong ones")
	flagUnseen   = flag.Int("unseen", 0, "if >0, this many options are never shown in train and dev")
	flagEvalFull = flag.Int("eval_full", 0, "with -candidates: also score this many test utterances against all the options")
	flagCacheDir = flag.String("cache_dir", "", "directory where the features are kept between runs (none if empty)")
	flagEncoder  = flag.String("encoder", "gemma", "gemma (the frozen Gemma 3 270M, mean of the hidden states) or embgemma2 (EmbeddingGemma 2, classification prompt)")
)

const usage = `Usage: go run ./cmd/intent [flags]

Assigns MASSIVE ja-JP utterances to functions on a frozen encoder: with a trained linear layer (fixed), or without training by the cosine between the utterance and the description of each function (cosine).

Examples (the commands of the article):
  # the first classifier: the 10 functions, a linear layer on the mean hidden state of the utterance
  go run ./cmd/intent -labels scenario10 -methods fixed -cache_dir ./cache
  # 60 intents, 12 options per utterance, 12 intents never seen in training; no training at all
  go run ./cmd/intent -labels intent60 -candidates 12 -unseen 12 -eval_full 1000 -methods cosine -cache_dir ./cache
  # the same with EmbeddingGemma 2
  go run ./cmd/intent -encoder embgemma2 -labels intent60 -candidates 12 -unseen 12 -eval_full 1000 -methods cosine -cache_dir ./cache

Flags:
`

// The settings of the article's runs.
const (
	dataDir   = "data/massive"
	epochs    = 30
	batchSize = 32 // training
	runs      = 3  // training runs per learning rate
	encBatch  = 64 // batch size of the frozen encoder when computing the features
	seed      = 1  // the option orders and distractors, the unseen options and the shuffles
)

// lrs are the Adam learning rates the fixed classifier tries.
var lrs = []float64{1e-3, 3e-3}

func must(err error) {
	if err != nil {
		log.Fatalf("%+v", err)
	}
}
func must1[T any](v T, err error) T              { must(err); return v }
func must2[A, B any](a A, b B, err error) (A, B) { must(err); return a, b }

type example struct {
	feature int   // the utterance's row in the feature bank
	opts    []int // the canonical index of each option, in the order shown
	gold    int   // the right option's position in opts, as scored
	canon   int   // the right option's canonical index, the Dense output trained to win
}

type split struct {
	name string
	exs  []example
}

func utterance(p massive.Prompt) string {
	s := strings.TrimPrefix(p.Text, "STATE:\n")
	u, _, ok := strings.Cut(s, "\nQUESTION:\n")
	if !ok {
		log.Fatalf("unexpected prompt %q", p.Text)
	}
	return u
}

func main() {
	flag.Usage = func() {
		fmt.Fprint(flag.CommandLine.Output(), usage)
		flag.PrintDefaults()
	}
	flag.Parse()
	t0 := time.Now()
	backend := compute.MustNew()
	utts := must1(massive.Load(must1(massive.EnsureJA(dataDir))))
	bb := must1(backbone.Load(backend, backbone.DefaultRepoID))
	sch, label := massive.Scenarios10(), massive.Scenario
	if *flagLabels == "intent60" {
		sch, label = massive.Intents60(), massive.Intent
	}
	var unseen []int
	if *flagUnseen > 0 {
		unseen = massive.PickUnseen(sch, *flagUnseen, seed)
		var names []string
		for _, c := range unseen {
			names = append(names, sch.Options[c].ID)
		}
		fmt.Printf("unseen options (never in train / dev): %s\n", strings.Join(names, ", "))
	}
	opts := massive.SplitOptions{Seed: seed, Candidates: *flagCands, Unseen: unseen}
	sp := must1(massive.BuildSplitsWith(bb.Tok, sch, utts, label, opts))
	var full []massive.Prompt
	if *flagEvalFull > 0 {
		fullOpts := opts
		fullOpts.Limit, fullOpts.Candidates = *flagEvalFull, 0
		full = must1(massive.BuildSplitsWith(bb.Tok, sch, utts, label, fullOpts)).Test
	}
	fmt.Printf("schema=%s | train=%d dev=%d test=%d (options shown: %d) | full-schema test=%d\n",
		sch.Name, len(sp.Train), len(sp.Dev), len(sp.Test), len(sp.Train[0].Order), len(full))

	H := bb.HiddenDim
	if *flagEncoder == "embgemma2" {
		H = 768
	}
	fb, optFB := newFeatureBank(H), newFeatureBank(H)
	var pending, optPending []item
	optItem := func(c int) item {
		pre := "OPTION:\n"
		line := sch.Options[c].ID + ": " + sch.Options[c].Desc
		return item{pre + line, len(pre), len(pre) + len(line)}
	}
	for c := range sch.Options {
		optFB.add(strconv.Itoa(c), optItem(c), &optPending)
	}
	convert := func(name string, exs []massive.Prompt) split {
		out := split{name: name}
		for _, ex := range exs {
			u := utterance(ex)
			e := example{gold: ex.Gold, canon: ex.Canon, opts: ex.Order}
			pre := "STATE:\n"
			e.feature = fb.add(u, item{pre + u, len(pre), len(pre) + len(u)}, &pending)
			out.exs = append(out.exs, e)
		}
		return out
	}
	trainSet, dev, test := convert("train", sp.Train), convert("dev", sp.Dev), convert("test", sp.Test)
	var fullS *split
	if full != nil {
		s := convert("test-all", full)
		fullS = &s
	}
	tag := fmt.Sprintf("%s-u%d-c%d-f%d", sch.Name, *flagUnseen, *flagCands, *flagEvalFull)
	switch *flagEncoder {
	case "gemma":
		exec := model.MustNewExec(backend, bb.Store, func(scope *model.Scope, tokens *Node) *Node { return bb.HiddenStates(scope, tokens) })
		optFB.fill(exec, bb.Tok, optPending, "option-"+sch.Name)
		fb.fill(exec, bb.Tok, pending, "utt-"+tag)
	case "embgemma2":
		em := must1(embgemma.Load(backend, embgemma.Repo))
		// The model card's prompt for classification, the same on both sides.
		optFB.fillEmbeddings(backend, em, optPending, embgemma.PromptClassification, "eg2opt-"+sch.Name)
		fb.fillEmbeddings(backend, em, pending, embgemma.PromptClassification, "eg2utt-"+tag)
	default:
		log.Fatalf("unknown -encoder %q", *flagEncoder)
	}

	classOf := map[int]int{}
	for c := range sch.Options {
		if !slices.Contains(unseen, c) {
			classOf[c] = len(classOf)
		}
	}

	var names []string
	for _, o := range sch.Options {
		names = append(names, o.ID)
	}
	ctx := &runCtx{backend: backend, fb: fb, optFB: optFB, classOf: classOf, unseen: unseen, names: names}
	type key struct {
		method string
		lr     float64
	}
	results := map[key][]evalRes{}
	var keys []key
	for m := range strings.SplitSeq(*flagMethods, ",") {
		m = strings.TrimSpace(m)
		mLRs, mRuns := lrs, runs
		if m == "cosine" {
			mLRs, mRuns = []float64{0}, 1
		}
		for _, lr := range mLRs {
			k := key{m, lr}
			keys = append(keys, k)
			for run := range mRuns {
				r := ctx.run(m, lr, run, trainSet, dev, test, fullS)
				results[k] = append(results[k], r)
				fmt.Printf("RESULT labels=%s unseen=%d method=%s lr=%g run=%d params=%d dev=%.4f test=%.4f known=%.4f unseen=%s full_known=%s full_unseen=%s secs=%.0f\n",
					sch.Name, len(unseen), m, lr, run, r.params, r.dev, r.test, r.known, fmtAcc(r.unseen), fmtAcc(r.fullKnown), fmtAcc(r.fullUnseen), r.secs)
			}
		}
	}
	fmt.Printf("\n================ SUMMARY (%s, unseen=%d, %d run(s), %d epochs) ================\n", sch.Name, len(unseen), runs, epochs)
	for _, k := range keys {
		rs := results[k]
		f := func(g func(evalRes) float64) string {
			var xs []float64
			for _, r := range rs {
				if v := g(r); !math.IsNaN(v) {
					xs = append(xs, v)
				}
			}
			if len(xs) == 0 {
				return "   n/a       "
			}
			m, s := meanStd(xs)
			return fmt.Sprintf("%.4f±%.4f", m, s)
		}
		fmt.Printf("%-10s lr=%-6g params=%-7d dev=%s test=%s | known=%s unseen=%s | all-options known=%s unseen=%s\n", k.method, k.lr, rs[0].params,
			f(func(r evalRes) float64 { return r.dev }), f(func(r evalRes) float64 { return r.test }),
			f(func(r evalRes) float64 { return r.known }), f(func(r evalRes) float64 { return r.unseen }),
			f(func(r evalRes) float64 { return r.fullKnown }), f(func(r evalRes) float64 { return r.fullUnseen }))
	}
	fmt.Printf("total wall time %v\n", time.Since(t0).Round(time.Second))
}

func fmtAcc(x float64) string {
	if math.IsNaN(x) {
		return "n/a"
	}
	return fmt.Sprintf("%.4f", x)
}

func meanStd(xs []float64) (m, s float64) {
	for _, x := range xs {
		m += x
	}
	m /= float64(len(xs))
	for _, x := range xs {
		s += (x - m) * (x - m)
	}
	return m, math.Sqrt(s / float64(len(xs)))
}
