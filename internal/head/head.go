// Package head implements the Decision Heads that score options from the Transformer hidden states.
package head

import (
	"math"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
)

// Kind selects the head architecture.
type Kind string

// Head kinds.
const (
	Linear  Kind = "linear"
	Routing Kind = "routing"
)

// Config configures a head.
type Config struct {
	Kind       Kind
	Dim        int     // width of the scorer and of the routing space
	Heads      int     // attention heads of the Evidence Routing and of the Cross-field
	Dropout    float64 // scorer dropout, training only
	CrossField bool    // the questions of a request look at each other before they are scored
}

// Output is the result of a head.
type Output struct {
	Logits *Node // [batch, numOptions]
	Attn   *Node // Evidence Routing weights [batch, heads, numOptions, seqLen], nil for Linear
}

// OptionMask turns option token ranges spans [batch, numOptions, 2] (end exclusive) into a mask [batch, numOptions, seqLen].
func OptionMask(spans *Node, seqLen int) *Node {
	g := spans.Graph()
	pos := Iota(g, shapes.Make(dtypes.Int32, 1, 1, seqLen), 2)
	start := Slice(spans, AxisRange(), AxisRange(), AxisElem(0))
	end := Slice(spans, AxisRange(), AxisRange(), AxisElem(1))
	return LogicalAnd(GreaterOrEqual(pos, start), LessThan(pos, end))
}

// PoolOptions averages the hidden states [batch, seqLen, hidden] over each option's tokens: [batch, numOptions, hidden].
func PoolOptions(hidden, optMask *Node) *Node {
	m := ConvertDType(optMask, hidden.DType())
	sum := Einsum("bkt,bth->bkh", m, hidden)
	count := Max(ReduceAndKeep(m, ReduceSum, -1), OnesLike(ReduceAndKeep(m, ReduceSum, -1)))
	return Div(sum, count)
}

// Score computes the option logits from hidden [batch, seqLen, hiddenDim], spans [batch, numOptions, 2], route
// [batch, seqLen], the tokens the Evidence Routing may look at, and group [batch], the request of each row, for the
// Cross-field.
func Score(scope *model.Scope, cfg Config, hidden, route, spans, group *Node) Output {
	hidden = ConvertDType(hidden, dtypes.Float32)
	seqLen := hidden.Shape().Dimensions[1]
	optMask := OptionMask(spans, seqLen)
	// Hidden states have huge outliers, so the pooled options are normalized.
	pooled := PoolOptions(hidden, optMask)
	pn := norm.LayerNorm(scope.In("ln_pooled"), pooled, -1).Done()
	dropout := func(s *model.Scope, x *Node) *Node {
		if cfg.Dropout <= 0 {
			return x
		}
		return layers.DropoutStatic(s, x, cfg.Dropout)
	}

	var x, attn *Node // the options to score [batch, numOptions, dim], and the Evidence Routing weights
	switch cfg.Kind {
	case Linear:
		x = pn

	case Routing:
		B, K := pn.Shape().Dimensions[0], pn.Shape().Dimensions[1]
		heads := cfg.Heads
		dk := cfg.Dim / heads

		xn := norm.LayerNorm(scope.In("ln_tokens"), hidden, -1).Done()
		q := layers.Dense(scope.In("query"), pn, true, heads*dk)
		q = Reshape(q, B, K, heads, dk)
		k := Reshape(layers.Dense(scope.In("key"), xn, true, heads*dk), B, seqLen, heads, dk)
		v := Reshape(layers.Dense(scope.In("value"), xn, true, heads*dk), B, seqLen, heads, dk)

		scores := MulScalar(Einsum("bkhd,bthd->bhkt", q, k), 1/math.Sqrt(float64(dk)))
		tokMask := BroadcastToShape(Reshape(route, B, 1, 1, seqLen), scores.Shape())
		attn = MaskedSoftmax(scores, tokMask, -1)
		ctx := Einsum("bhkt,bthd->bkhd", attn, v)
		ctx = Reshape(ctx, B, K, heads*dk)
		evidence := layers.Dense(scope.In("evidence"), ctx, true, cfg.Dim)

		opt := layers.Dense(scope.In("option"), pn, true, cfg.Dim)
		z := activation.Gelu(layers.Dense(scope.In("fuse"), Concatenate([]*Node{opt, evidence}, -1), true, cfg.Dim))
		x = dropout(scope.In("drop"), z)

	default:
		panic("unknown head " + string(cfg.Kind))
	}
	if cfg.CrossField {
		x = crossField(scope.In("cross"), cfg, x, optMask, group)
	}
	return Output{Logits: Squeeze(layers.Dense(scope.In("scorer"), x, true, 1), -1), Attn: attn}
}

// crossField lets each question of a request look at the other questions of the same request, the rows of the batch
// with the same group, and adds what it saw to its options x [batch, numOptions, dim].
func crossField(scope *model.Scope, cfg Config, x, optMask, group *Node) *Node {
	B, K := x.Shape().Dimensions[0], x.Shape().Dimensions[1]
	D, heads := cfg.Dim, cfg.Heads
	dk := D / heads
	if x.Shape().Dimensions[2] != D {
		x = layers.Dense(scope.In("in"), x, true, D)
	}
	// a question is the mean of its options
	valid := ReduceAndKeep(ConvertDType(optMask, x.DType()), ReduceMax, -1) // [B, K, 1]
	count := Max(ReduceSum(valid, 1), OnesLike(ReduceSum(valid, 1)))
	u := Div(ReduceSum(Mul(x, valid), 1), count) // [B, D]

	un := norm.LayerNorm(scope.In("ln_attn"), u, -1).Done()
	q := Reshape(layers.Dense(scope.In("query"), un, true, D), B, heads, dk)
	k := Reshape(layers.Dense(scope.In("key"), un, true, D), B, heads, dk)
	v := Reshape(layers.Dense(scope.In("value"), un, true, D), B, heads, dk)
	scores := MulScalar(Einsum("ihd,jhd->hij", q, k), 1/math.Sqrt(float64(dk)))
	same := Equal(Reshape(group, 1, B, 1), Reshape(group, 1, 1, B))
	attn := MaskedSoftmax(scores, BroadcastToShape(same, scores.Shape()), -1)
	ctx := Reshape(Einsum("hij,jhd->ihd", attn, v), B, D)
	u = Add(u, layers.Dense(scope.In("out"), ctx, true, D))
	ff := activation.Gelu(layers.Dense(scope.In("ff1"), norm.LayerNorm(scope.In("ln_ff"), u, -1).Done(), true, 2*D))
	u = Add(u, layers.Dense(scope.In("ff2"), ff, true, D))

	// the same u for every option would shift all their logits alike: it goes through a nonlinearity with each option
	uk := BroadcastToDims(Reshape(u, B, 1, D), B, K, D)
	return Add(x, activation.Gelu(layers.Dense(scope.In("fuse"), Concatenate([]*Node{x, uk}, -1), true, D)))
}
