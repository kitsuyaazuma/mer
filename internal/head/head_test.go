package head

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
)

func TestCrossFieldStaysInRequest(t *testing.T) {
	const B, T, H, K = 4, 6, 8, 2
	cfg := Config{Kind: Routing, Dim: 16, Heads: 2, CrossField: true}
	store := model.NewStore()
	store.SetParam(model.ParamInitialSeed, int64(1))
	exec := model.MustNewExec(compute.MustNew(), store, func(scope *model.Scope, hidden, route, spans, group *Node) *Node {
		return Score(scope.In("head"), cfg, hidden, route, spans, group).Logits
	})
	route := make([]bool, B*T)
	for i := range route {
		route[i] = true
	}
	spans := make([]int32, 0, B*K*2)
	for range B {
		spans = append(spans, 2, 4, 4, 6)
	}
	group := []int32{0, 0, 1, -1} // rows 0 and 1 are one request, row 2 another, row 3 padding
	logits := func(row int, v float32) []float32 {
		hidden := make([]float32, B*T*H)
		for i := range hidden {
			hidden[i] = float32(math.Sin(float64(i)))
		}
		// a different change per feature: a uniform one is undone by the LayerNorm
		for i := row * T * H; i < (row+1)*T*H; i++ {
			hidden[i] += v * float32(math.Cos(1.7*float64(i)))
		}
		out := exec.MustCall1(tensors.FromFlatDataAndDimensions(hidden, B, T, H), tensors.FromFlatDataAndDimensions(route, B, T),
			tensors.FromFlatDataAndDimensions(spans, B, K, 2), tensors.FromFlatDataAndDimensions(group, B))
		return tensors.MustCopyFlatData[float32](out)
	}
	base, moved := logits(1, 0), logits(1, 1) // row 1 changes
	if base[0] == moved[0] {
		t.Errorf("row 0 does not see row 1 of its request: %v", base[0])
	}
	if base[4] != moved[4] || base[5] != moved[5] {
		t.Errorf("row 2 sees row 1 of another request: %v %v", base[4:6], moved[4:6])
	}
}
