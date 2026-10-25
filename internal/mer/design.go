package mer

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/peft"
	"github.com/gomlx/gomlx/ml/model"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/cache"
	"github.com/kitsuyaazuma/mer/internal/head"
)

// Design is the head that scores the criteria: linear, routing (Evidence Routing over every token) or routing_state
// (over the state only), each with the Cross-field when its name ends in _cross.
type Design struct {
	Name       string
	Head       head.Kind
	StateOnly  bool
	CrossField bool
}

// ParseDesign reads a design name.
func ParseDesign(name string) (Design, error) {
	d := Design{Name: name}
	base, cross := strings.CutSuffix(name, "_cross")
	d.CrossField = cross
	switch base {
	case "linear":
		d.Head = head.Linear
	case "routing":
		d.Head = head.Routing
	case "routing_state":
		d.Head, d.StateOnly = head.Routing, true
	default:
		return d, fmt.Errorf("unknown design %q (linear | routing | routing_state, each with or without _cross)", name)
	}
	return d, nil
}

// HeadConfig is the head of the design.
func (d Design) HeadConfig() head.Config {
	if d.Head == head.Routing {
		return head.Config{Kind: head.Routing, Dim: 256, Heads: 4, Dropout: 0.1, CrossField: d.CrossField}
	}
	return head.Config{Kind: head.Linear, Dim: 256, Heads: 4, CrossField: d.CrossField}
}

// LR is the learning rate of the head of the design on the frozen backbone.
func (d Design) LR() float64 {
	if d.Head == head.Routing || d.CrossField {
		return 3e-4
	}
	return 3e-3
}

// HeadExec scores the criteria from given hidden states with the head kept in store.
func (d Design) HeadExec(backend compute.Backend, store *model.Store) *model.Exec {
	cfg := d.HeadConfig()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, route, spans, group, hidden *Node) *Node {
		return head.Score(scope.In("head"), cfg, hidden, route, spans, group).Logits
	})
	exec.SetMaxCache(512)
	return exec
}

// LoRAExec scores the criteria with the backbone, its LoRA adapters and the head, all kept in bb.Store.
func (d Design) LoRAExec(backend compute.Backend, bb *backbone.Backbone) *model.Exec {
	cfg := d.HeadConfig()
	exec := model.MustNewExec(backend, bb.Store, func(scope *model.Scope, tokens, spans, route, group *Node) *Node {
		return head.Score(scope.In("head"), cfg, bb.HiddenStates(scope, tokens), route, spans, group).Logits
	})
	exec.SetMaxCache(512)
	return exec
}

// routeAndSpans are the tokens the routing looks at, the criterion spans and the request (the same number for the
// questions of one request, -1 for a padding row) of the examples idx of exs, for a batch of B sequences of T tokens
// and K criteria.
func (d Design) routeAndSpans(exs []Example, idx []int, B, T, K int) (route []bool, spans, group []int32) {
	route = make([]bool, B*T)
	spans = make([]int32, B*K*2)
	group = make([]int32, B)
	for r := range group {
		group[r] = -1
	}
	for r, i := range idx {
		e := &exs[i]
		group[r] = int32(e.Parent)
		lo, hi := 0, len(e.IDs)
		if d.StateOnly {
			lo, hi = int(e.StateSpan[0]), int(e.StateSpan[1])
		}
		for t := lo; t < hi; t++ {
			route[r*T+t] = true
		}
		for k, sp := range e.Spans {
			spans[(r*K+k)*2], spans[(r*K+k)*2+1] = sp[0], sp[1]
		}
	}
	return route, spans, group
}

// FrozenInputs are the inputs of HeadExec for the examples idx of exs: the routed tokens, the criterion spans, the
// requests and the cached hidden states.
func (d Design) FrozenInputs(exs []Example, c *cache.VarCache, idx []int, B, T, K int) ([]*tensors.Tensor, error) {
	route, spans, group := d.routeAndSpans(exs, idx, B, T, K)
	full := slices.Clone(idx)
	for len(full) < B { // padding rows: any example, its loss weight is 0
		full = append(full, idx[0])
	}
	hidden, err := c.Gather(full, T)
	if err != nil {
		return nil, err
	}
	return []*tensors.Tensor{tensors.FromFlatDataAndDimensions(route, B, T), tensors.FromFlatDataAndDimensions(spans, B, K, 2),
		tensors.FromFlatDataAndDimensions(group, B), hidden}, nil
}

// LoRAInputs are the inputs of LoRAExec for the examples idx of exs: the tokens, the criterion spans, the routed
// tokens and the requests.
func (d Design) LoRAInputs(exs []Example, pad int32, idx []int, B, T, K int) []*tensors.Tensor {
	route, spans, group := d.routeAndSpans(exs, idx, B, T, K)
	toks := padded(exs, idx, B, T, pad)
	return []*tensors.Tensor{tensors.FromFlatDataAndDimensions(toks, B, T), tensors.FromFlatDataAndDimensions(spans, B, K, 2),
		tensors.FromFlatDataAndDimensions(route, B, T), tensors.FromFlatDataAndDimensions(group, B)}
}

// Inputs builds the inputs of one batch of a prediction.
type Inputs func(idx []int, B, T, K int) ([]*tensors.Tensor, error)

// Predict runs exec over exs, batching examples of similar length, and returns the criterion logits of each. With
// whole, the questions of a request go into one batch, as the Cross-field needs.
func Predict(exec *model.Exec, exs []Example, whole bool, inputs Inputs) ([][]float32, error) {
	out := make([][]float32, len(exs))
	err := batches(exs, whole, func(idx []int, B, T int) error {
		K := 0
		for _, i := range idx {
			K = max(K, exs[i].K())
		}
		K = Bucket(K)
		in, err := inputs(idx, B, T, K)
		if err != nil {
			return err
		}
		args := make([]any, len(in))
		for i, t := range in {
			args[i] = t
			defer func() { _ = t.FinalizeAll() }()
		}
		res, err := exec.Call1(args...)
		if err != nil {
			return err
		}
		defer func() { _ = res.FinalizeAll() }()
		flat, err := tensors.CopyFlatData[float32](res)
		if err != nil {
			return err
		}
		for r, i := range idx {
			out[i] = slices.Clone(flat[r*K : (r+1)*K])
		}
		return nil
	})
	return out, err
}

// SetLoRA puts rank-8 LoRA adapters on the query and value projections of every backbone layer; the head is trained
// whole.
func SetLoRA(bb *backbone.Backbone) {
	bb.Store.SetParams(map[string]any{
		peft.ParamAdapter:       "lora",
		peft.ParamRank:          8,
		peft.ParamAlpha:         float32(16),
		peft.ParamDropout:       float32(0.05),
		peft.ParamTargetModules: "query,value",
	})
	bb.Store.RootScope().In("head").SetParam(peft.ParamAdapter, "")
}
