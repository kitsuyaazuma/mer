package main

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/head"
	"github.com/kitsuyaazuma/mer/internal/mer"
)

type trainer struct {
	backend compute.Backend
	bb      *backbone.Backbone
}

// predictor returns the criterion logits of every sequence of a split.
type predictor func(s *split) [][]float32

// loss is the cross entropy of each sequence's question over its own criteria, averaged over the sequences.
func loss(labels, predictions []*Node) *Node {
	lg, gold, valid := predictions[0], labels[0], labels[1]
	K := lg.Shape().Dimensions[1]
	masked := Where(valid, lg, AddScalar(ZerosLike(lg), -1e9))
	mx := StopGradient(ReduceAndKeep(masked, ReduceMax, -1))
	lse := Add(Squeeze(mx, -1), Log(ReduceSum(Exp(Sub(masked, mx)), -1)))
	oneHot := OneHot(Max(gold, ZerosLike(gold)), K, dtypes.Float32)
	nll := Sub(lse, ReduceSum(Mul(lg, oneHot), -1))
	asked := ConvertDType(GreaterOrEqual(gold, ZerosLike(gold)), dtypes.Float32)
	count := ReduceAllSum(asked)
	return Div(ReduceAllSum(Mul(nll, asked)), Max(count, OnesLike(count)))
}

// batchLabels are the right criterion (-1 for a padding row) and the real criteria of each row.
func batchLabels(s *split, idx []int, B, K int) []*tensors.Tensor {
	gold := make([]int32, B)
	valid := make([]bool, B*K)
	for r := range gold {
		gold[r] = -1
	}
	for r, i := range idx {
		e := &s.exs[i]
		gold[r] = int32(e.Question().Gold)
		for k := range e.K() {
			valid[r*K+k] = true
		}
	}
	return []*tensors.Tensor{tensors.FromFlatDataAndDimensions(gold, B), tensors.FromFlatDataAndDimensions(valid, B, K)}
}

// units are what training batches are made of: the examples one by one, or with whole the questions of each request
// together.
func units(exs []mer.Example, whole bool) [][]int {
	order := make([]int, len(exs))
	for i := range order {
		order[i] = i
	}
	if whole {
		return mer.Requests(exs, order)
	}
	out := make([][]int, len(exs))
	for i := range out {
		out[i] = order[i : i+1]
	}
	return out
}

// unitLen is the length of the longest example of u.
func unitLen(exs []mer.Example, u []int) int {
	n := 0
	for _, i := range u {
		n = max(n, len(exs[i].IDs))
	}
	return n
}

// shuffled is us in a random order, then sorted by length within chunks of n, so that batches hold similar lengths.
func shuffled(exs []mer.Example, us [][]int, n int, rng *rand.Rand) [][]int {
	perm := rng.Perm(len(us))
	out := make([][]int, len(us))
	for i, p := range perm {
		out[i] = us[p]
	}
	for lo := 0; lo < len(out); lo += n {
		part := out[lo:min(lo+n, len(out))]
		sort.SliceStable(part, func(a, b int) bool { return unitLen(exs, part[a]) < unitLen(exs, part[b]) })
	}
	return out
}

// epochBatches cuts the shuffled units into batches of at most bs examples, or one unit when it is more.
func epochBatches(exs []mer.Example, us [][]int, bs int, rng *rand.Rand) [][]int {
	us = shuffled(exs, us, 32*bs, rng)
	var out [][]int
	for lo := 0; lo < len(us); {
		idx := slices.Clone(us[lo])
		for lo++; lo < len(us) && len(idx)+len(us[lo]) <= bs && lo%(32*bs) != 0; lo++ {
			idx = append(idx, us[lo]...)
		}
		out = append(out, idx)
	}
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

func roundUp(n, m int) int { return (n + m - 1) / m * m }

// maxCriteria is the number of criteria every training batch is padded to.
func maxCriteria(exs []mer.Example) int {
	K := 0
	for _, e := range exs {
		K = max(K, e.K())
	}
	return roundUp(K, 8)
}

// longest is the bucketed length of the longest sequence among idx.
func longest(exs []mer.Example, idx []int) int {
	T := 0
	for _, i := range idx {
		T = max(T, len(exs[i].IDs))
	}
	return mer.Bucket(T)
}

// train trains the head of d on the cached hidden states of the frozen backbone.
func (t *trainer) train(d mer.Design, run int, tr *split) (predict predictor, params int) {
	cfg := d.HeadConfig()
	store := model.NewStore()
	store.SetParam(model.ParamInitialSeed, int64(1000+run))
	modelFn := func(scope *model.Scope, _ any, in []*Node) []*Node {
		return []*Node{head.Score(scope.In("head"), cfg, in[3], in[0], in[1], in[2]).Logits}
	}
	trainer := train.NewTrainer(t.backend, store, modelFn, loss, optimizer.Adam().LearningRate(d.LR()).Done(), []metric.Interface{}, []metric.Interface{})
	K := maxCriteria(tr.exs)
	us := units(tr.exs, d.CrossField)
	rng := rand.New(rand.NewSource(seed + 7919*int64(run)))
	for range epochs {
		for _, idx := range epochBatches(tr.exs, us, batchSize, rng) {
			T, B := longest(tr.exs, idx), max(batchSize, roundUp(len(idx), 8))
			batch := train.Batch{Inputs: must1(d.FrozenInputs(tr.exs, tr.cache, idx, B, T, K)), Labels: batchLabels(tr, idx, B, K)}
			must1(trainer.TrainStep(batch))
			_ = batch.Finalize()
		}
	}
	params, _ = backbone.ParamCounts(store)
	if *flagSave != "" && run == 0 {
		save(store, d)
	}
	exec := d.HeadExec(t.backend, store)
	return func(s *split) [][]float32 {
		return must1(mer.Predict(exec, s.exs, d.CrossField, func(idx []int, B, T, K int) ([]*tensors.Tensor, error) {
			return d.FrozenInputs(s.exs, s.cache, idx, B, T, K)
		}))
	}, params
}

// loraBatches cuts the shuffled units into batches of at most loraBudget tokens, or one unit when it is more.
func loraBatches(exs []mer.Example, us [][]int, rng *rand.Rand) [][]int {
	us = shuffled(exs, us, 512, rng)
	var out [][]int
	for lo := 0; lo < len(us); {
		idx, T := slices.Clone(us[lo]), unitLen(exs, us[lo])
		for lo++; lo < len(us) && lo%512 != 0; lo++ {
			t := max(T, unitLen(exs, us[lo]))
			if (len(idx)+len(us[lo]))*mer.Bucket(t) > loraBudget {
				break
			}
			idx, T = append(idx, us[lo]...), t
		}
		out = append(out, idx)
	}
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// trainLoRA trains the LoRA adapters of the backbone together with the head of d.
func (t *trainer) trainLoRA(d mer.Design, run int, tr *split) (predict predictor, params int) {
	bb := t.bb
	// without a seed GoMLX draws a fresh one per process (LoRA A, head, dropout), and no two runs would match
	bb.Store.SetParam(model.ParamInitialSeed, int64(1000+run))
	must(bb.Store.SetRNGStateFromSeed(int64(1000 + run)))
	const epochs = 1
	lr := *flagLoRALR
	if *flagClip > 0 {
		bb.Store.SetParam(optimizer.ParamClipStepByValue, *flagClip)
	}
	steps := 0 // of the whole training, known before the first step builds the graph
	cfg := d.HeadConfig()
	modelFn := func(scope *model.Scope, _ any, in []*Node) []*Node {
		if *flagWarmup > 0 || *flagCosine {
			period := 1 << 30 // after the warmup the learning rate stays at lr
			if *flagCosine {
				period = max(1, steps-*flagWarmup)
			}
			cosineschedule.New(scope, in[0].Graph(), dtypes.Float32).LearningRate(lr).WarmUpSteps(*flagWarmup).PeriodSteps(period).Done()
		}
		hidden := bb.HiddenStates(scope, in[0])
		return []*Node{head.Score(scope.In("head"), cfg, hidden, in[2], in[1], in[3]).Logits}
	}
	trainer := train.NewTrainer(t.backend, bb.Store, modelFn, loss, optimizer.Adam().LearningRate(lr).Done(), []metric.Interface{}, []metric.Interface{})
	K := maxCriteria(tr.exs)
	rng := rand.New(rand.NewSource(seed))
	t0 := time.Now()
	var stepTimes []time.Duration
	us := units(tr.exs, d.CrossField)
	fmt.Printf("  lora: lr %g, warmup %d, cosine %v, clip %g, max_steps %d\n", lr, *flagWarmup, *flagCosine, *flagClip, *flagMaxSteps)
	for epoch := range epochs {
		batches := loraBatches(tr.exs, us, rng)
		if *flagMaxSteps > 0 && len(batches) > *flagMaxSteps {
			batches = batches[:*flagMaxSteps]
		}
		steps = epochs * len(batches)
		biggest := 0
		for _, idx := range batches {
			biggest = max(biggest, len(idx)*longest(tr.exs, idx))
		}
		fmt.Printf("  lora epoch %d: %d batches, the largest %d tokens padded\n", epoch+1, len(batches), biggest)
		lossSum, winSum := 0.0, 0.0
		for bi, idx := range batches {
			T := longest(tr.exs, idx)
			B := max(1, loraBudget/T)
			for B > 1 && B*T > loraBudget {
				B--
			}
			B = max(B, len(idx))
			ts := time.Now()
			batch := train.Batch{Inputs: d.LoRAInputs(tr.exs, bb.PadID, idx, B, T, K), Labels: batchLabels(tr, idx, B, K)}
			m := must1(trainer.TrainStep(batch))
			_ = batch.Finalize()
			stepTimes = append(stepTimes, time.Since(ts))
			l := float64(tensors.ToScalar[float32](m[0]))
			lossSum, winSum = lossSum+l, winSum+l
			if (bi+1)%100 == 0 {
				fmt.Printf("  lora epoch %d step %d/%d loss(last 100) %.4f loss(avg) %.4f lr %.2e | %v elapsed\n", epoch+1, bi+1, len(batches),
					winSum/100, lossSum/float64(bi+1), learningRate(bb.Store), time.Since(t0).Round(time.Second))
				winSum = 0
			}
		}
	}
	slices.Sort(stepTimes)
	trainable, frozen := backbone.ParamCounts(bb.Store)
	fmt.Printf("  lora: trainable %d, frozen %d, %d steps, median step %v, training %v, last lr %.2e\n", trainable, frozen, len(stepTimes),
		stepTimes[len(stepTimes)/2].Round(time.Millisecond), time.Since(t0).Round(time.Second), learningRate(bb.Store))
	if *flagSave != "" {
		save(bb.Store, d)
	}
	if d.CrossField {
		// a request can have more questions than fit one batch of the backbone: the hidden states come first
		hidden, exec := mer.HiddenExec(t.backend, bb), d.HeadExec(t.backend, bb.Store)
		return func(s *split) [][]float32 {
			c := must1(mer.HiddenStates(hidden, bb, s.exs, "", ""))
			defer func() { _ = c.Close() }()
			return must1(mer.Predict(exec, s.exs, true, func(idx []int, B, T, K int) ([]*tensors.Tensor, error) {
				return d.FrozenInputs(s.exs, c, idx, B, T, K)
			}))
		}, trainable
	}
	exec := d.LoRAExec(t.backend, bb)
	return func(s *split) [][]float32 {
		return must1(mer.Predict(exec, s.exs, false, func(idx []int, B, T, K int) ([]*tensors.Tensor, error) {
			return d.LoRAInputs(s.exs, bb.PadID, idx, B, T, K), nil
		}))
	}, trainable
}

// learningRate is the current learning rate of the optimizer in store.
func learningRate(store *model.Store) float64 {
	v := store.GetVariable("/" + optimizer.Scope + "/" + optimizer.ParamLearningRate)
	if v == nil {
		return math.NaN()
	}
	return float64(tensors.ToScalar[float32](v.MustValue()))
}

// save writes the trained weights of d to -save.
func save(store *model.Store, d mer.Design) {
	must(mer.Save(*flagSave, store, mer.Meta{Design: d.Name, Backbone: *flagBackbone, DType: *flagDType, LoRA: *flagLoRA}))
	fmt.Printf("  saved %s to %s\n", d.Name, *flagSave)
}
