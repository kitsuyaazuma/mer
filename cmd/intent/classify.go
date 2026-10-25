package main

import (
	"log"
	"math"
	"math/rand"
	"slices"
	"time"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	"github.com/kitsuyaazuma/mer/internal/backbone"
)

type runCtx struct {
	backend   compute.Backend
	fb, optFB *featureBank
	classOf   map[int]int
	unseen    []int
	names     []string
}

type evalRes struct {
	params                   int
	dev, test, known, unseen float64 // accuracies; known and unseen split test by whether training saw the right option
	fullKnown, fullUnseen    float64 // known and unseen against all the options (-eval_full)
	secs                     float64 // how long the run took, training and scoring
}

func layerNorm(scope *model.Scope, x *Node) *Node { return norm.LayerNorm(scope, x, -1).Done() }

func fixedLogits(scope *model.Scope, u *Node, classes int) *Node {
	return layers.Dense(scope.In("classes"), layerNorm(scope.In("ln"), u), true, classes)
}

func (c *runCtx) inputs(exs []example, idx []int) *tensors.Tensor {
	flat := make([]float32, 0, len(idx)*c.fb.H)
	for _, i := range idx {
		flat = append(flat, c.fb.rows[exs[i].feature]...)
	}
	return tensors.FromFlatDataAndDimensions(flat, len(idx), c.fb.H)
}

func (c *runCtx) run(method string, lr float64, run int, trainSet, dev, test split, full *split) evalRes {
	t0 := time.Now()
	res := evalRes{fullKnown: math.NaN(), fullUnseen: math.NaN()}
	switch method {
	case "fixed":
		head, params := c.trainFixed(lr, run, trainSet)
		res.params = params
		res.dev, _, _ = c.evaluateFixed(head, dev)
		res.test, res.known, res.unseen = c.evaluateFixed(head, test)
		if full != nil {
			_, res.fullKnown, res.fullUnseen = c.evaluateFixed(head, *full)
		}
	case "cosine":
		res.dev, _, _ = c.evaluateCosine(dev)
		res.test, res.known, res.unseen = c.evaluateCosine(test)
		if full != nil {
			_, res.fullKnown, res.fullUnseen = c.evaluateCosine(*full)
		}
	default:
		log.Fatalf("unknown method %q (fixed | cosine)", method)
	}
	res.secs = time.Since(t0).Seconds()
	return res
}

// trainFixed trains the fixed-class Head on trainSet and returns it, compiled for scoring, with its number of parameters.
func (c *runCtx) trainFixed(lr float64, run int, trainSet split) (*model.Exec, int) {
	// A new Store holds only the Head: the features were computed once, so Gemma is not part of training.
	// Its seed makes every run start from the same initial Head.
	store := model.NewStore()
	store.SetParam(model.ParamInitialSeed, int64(1000+run))
	modelFn := func(scope *model.Scope, _ any, in []*Node) []*Node {
		return []*Node{fixedLogits(scope.In("head"), in[0], len(c.classOf))}
	}
	trainer := train.NewTrainer(c.backend, store, modelFn, loss.SparseCategoricalCrossEntropyLogits,
		optimizer.Adam().LearningRate(lr).Done(),
		[]metric.Interface{metric.NewMovingAverageSparseCategoricalAccuracy("Moving Accuracy", "~acc", 0.05)},
		[]metric.Interface{metric.NewSparseCategoricalAccuracy("Accuracy", "acc")})
	n := len(trainSet.exs)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	rng := rand.New(rand.NewSource(seed + 7919*int64(run)))
	for range epochs {
		rng.Shuffle(n, func(i, j int) { order[i], order[j] = order[j], order[i] })
		for lo := 0; lo < n; lo += batchSize {
			idx := order[lo:min(lo+batchSize, n)]
			labels := make([]int32, len(idx))
			for i, e := range idx {
				labels[i] = int32(c.classOf[trainSet.exs[e].canon]) // the Dense output of the right function, not its shown position
			}
			batch := train.Batch{Inputs: []*tensors.Tensor{c.inputs(trainSet.exs, idx)},
				Labels: []*tensors.Tensor{tensors.FromFlatDataAndDimensions(labels, len(idx), 1)}}
			must1(trainer.TrainStep(batch))
			_ = batch.Finalize()
		}
	}
	params, _ := backbone.ParamCounts(store)
	head := model.MustNewExec(c.backend, store, func(scope *model.Scope, x *Node) *Node {
		return fixedLogits(scope.In("head"), x, len(c.classOf))
	})
	return head, params
}

// evaluateFixed returns the accuracy of the trained Head over all of s, over its examples of seen options, and over those of unseen options.
func (c *runCtx) evaluateFixed(head *model.Exec, s split) (all, known, unseenAcc float64) {
	hits := make([]bool, len(s.exs))
	for lo := 0; lo < len(s.exs); lo += 256 {
		idx := make([]int, 0, 256)
		for i := lo; i < min(lo+256, len(s.exs)); i++ {
			idx = append(idx, i)
		}
		in := c.inputs(s.exs, idx)
		out := head.MustCall1(in)
		flat := tensors.MustCopyFlatData[float32](out)
		_ = out.FinalizeAll()
		_ = in.FinalizeAll()
		W := len(flat) / len(idx)
		for i, e := range idx {
			// The logits follow the functions' canonical order; score the shown options, in the order gold counts in.
			row := flat[i*W : (i+1)*W]
			var scores []float32
			for _, o := range s.exs[e].opts {
				scores = append(scores, row[c.classOf[o]])
			}
			hits[e] = argmax(scores) == s.exs[e].gold
		}
	}
	return c.accuracies(s, hits)
}

// evaluateCosine returns the accuracies of picking the shown option whose description is the most similar to the utterance, without training.
func (c *runCtx) evaluateCosine(s split) (all, known, unseenAcc float64) {
	hits := make([]bool, len(s.exs))
	for i, e := range s.exs {
		u := c.fb.rows[e.feature]
		var scores []float32
		for _, o := range e.opts {
			scores = append(scores, cosine(u, c.optFB.rows[o]))
		}
		hits[i] = argmax(scores) == e.gold
	}
	return c.accuracies(s, hits)
}

// accuracies splits the hits of s into all its examples, those of seen options and those of unseen options.
func (c *runCtx) accuracies(s split, hits []bool) (all, known, unseenAcc float64) {
	var ok, okK, nK, okU, nU int
	for i, e := range s.exs {
		isUnseen := slices.Contains(c.unseen, e.canon)
		if isUnseen {
			nU++
		} else {
			nK++
		}
		if hits[i] {
			ok++
			if isUnseen {
				okU++
			} else {
				okK++
			}
		}
	}
	div := func(a, b int) float64 {
		if b == 0 {
			return math.NaN()
		}
		return float64(a) / float64(b)
	}
	return div(ok, len(s.exs)), div(okK, nK), div(okU, nU)
}

func cosine(a, b []float32) float32 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return float32(ab / math.Sqrt(aa*bb))
}

func argmax(x []float32) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}
