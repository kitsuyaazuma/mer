// Package embgemma runs the EmbeddingGemma 2 text encoder in GoMLX to produce 768-dimensional sentence vectors.
package embgemma

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/safetensors"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
)

// Repo is the (Apache 2.0, un-gated) EmbeddingGemma 2 checkpoint.
const Repo = "google/embeddinggemma-2"

// Prompts recommended by the model card (config_sentence_transformers.json).
const (
	PromptClassification = "task: classification | query: "
	PromptQuery          = "task: search result | query: "
	PromptDocument       = "title: none | text: "
)

type textConfig struct {
	HiddenSize       int      `json:"hidden_size"`
	EmbeddingDim     int      `json:"embedding_dim"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`
	PerLayerInput    int      `json:"hidden_size_per_layer_input"`
	RMSNormEps       float64  `json:"rms_norm_eps"`
	SlidingWindow    int      `json:"sliding_window"`
	LayerTypes       []string `json:"layer_types"`
	PerLayerOverride map[string]struct {
		HeadDim    int `json:"head_dim"`
		NumKVHeads int `json:"num_key_value_heads"`
	} `json:"per_layer_config"`
	RoPE map[string]struct {
		Theta float64 `json:"rope_theta"`
	} `json:"rope_parameters"`
}

// Model is the loaded text encoder.
type Model struct {
	Tok   tokenizers.Tokenizer
	Store *model.Store
	cfg   textConfig
	vars  map[string]*model.Variable
}

// Load reads the configuration, the tokenizer and the text weights (converted to float32) of repoID.
func Load(backend compute.Backend, repoID string) (*Model, error) {
	repo := hub.New(repoID)
	if err := repo.DownloadInfo(false); err != nil {
		return nil, err
	}
	files, err := repo.DownloadFiles("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(files[0])
	if err != nil {
		return nil, err
	}
	var top struct {
		Text textConfig `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("parsing config.json: %w", err)
	}
	m := &Model{cfg: top.Text, Store: model.NewStore(), vars: map[string]*model.Variable{}}
	if m.Tok, err = tokenizers.New(repo); err != nil {
		return nil, err
	}
	if err := m.Tok.With(api.EncodeOptions{AddSpecialTokens: true}); err != nil {
		return nil, err
	}
	st, err := safetensors.New(repo)
	if err != nil {
		return nil, err
	}
	for _, name := range st.ListTensorNames() {
		short, ok := strings.CutPrefix(name, "language_model.")
		if !ok {
			continue // vision / audio encoders
		}
		t, err := st.GetTensor(backend, name)
		if err != nil {
			return nil, err
		}
		v := t.Tensor
		if v.DType() != dtypes.Float32 {
			if v, err = CallOnce(backend, func(x *Node) *Node { return ConvertDType(x, dtypes.Float32) }, v); err != nil {
				return nil, fmt.Errorf("converting %s: %w", name, err)
			}
		}
		path := strings.ReplaceAll(short, ".", "_")
		m.vars[short] = m.Store.RootScope().VariableWithValue(path, v)
		m.vars[short].SetTrainable(false)
	}
	if len(m.vars) == 0 {
		return nil, errors.New("no language_model.* tensor in the checkpoint")
	}
	return m, nil
}

// Encode tokenizes a text as sentence-transformers does: <bos> text <eos>.
func (m *Model) Encode(text string) []int32 {
	ids := m.Tok.Encode(text)
	out := make([]int32, 0, len(ids)+1)
	for _, id := range ids {
		out = append(out, int32(id))
	}
	if eos, err := m.Tok.SpecialTokenID(api.TokEndOfSentence); err == nil && (len(out) == 0 || out[len(out)-1] != int32(eos)) {
		out = append(out, int32(eos))
	}
	return out
}

func (m *Model) layer(i int) (headDim, kvHeads int, theta float64) {
	headDim, kvHeads = m.cfg.HeadDim, m.cfg.NumKVHeads
	if o, ok := m.cfg.PerLayerOverride[fmt.Sprintf("%02d", i)]; ok {
		headDim, kvHeads = o.HeadDim, o.NumKVHeads
	}
	return headDim, kvHeads, m.cfg.RoPE[m.cfg.LayerTypes[i]].Theta
}

// rmsNorm multiplies by the weight itself, not 1+weight (Gemma 4 style).
func rmsNorm(x, w *Node, eps float64) *Node {
	ms := ReduceAndKeep(Square(x), ReduceMean, -1)
	x = Mul(x, Rsqrt(AddScalar(ms, eps)))
	if w != nil {
		dims := make([]int, x.Rank())
		for i := range dims {
			dims[i] = 1
		}
		dims[len(dims)-1] = w.Shape().Size()
		x = Mul(x, Reshape(w, dims...))
	}
	return x
}

func linear(x, w *Node) *Node {
	dims := x.Shape().Dimensions
	in, out := dims[len(dims)-1], w.Shape().Dimensions[0]
	y := Einsum("ni,oi->no", Reshape(x, -1, in), w)
	return Reshape(y, append(slices.Clone(dims[:len(dims)-1]), out)...)
}

// rope uses the rotate-half convention.
func rope(x *Node, theta float64) *Node {
	g := x.Graph()
	T, hd := x.Shape().Dimensions[1], x.Shape().Dimensions[3]
	half := hd / 2
	invFreq := make([]float32, half)
	for i := range invFreq {
		invFreq[i] = float32(1 / math.Pow(theta, float64(2*i)/float64(hd)))
	}
	pos := ConvertDType(Iota(g, shapes.Make(dtypes.Int32, T, 1), 0), dtypes.Float32)
	freqs := Mul(pos, Reshape(Const(g, invFreq), 1, half))
	emb := Concatenate([]*Node{freqs, freqs}, -1)
	cos, sin := Reshape(Cos(emb), 1, T, 1, hd), Reshape(Sin(emb), 1, T, 1, hd)
	x1, x2 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(0, half)), Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(half, hd))
	rot := Concatenate([]*Node{Neg(x2), x1}, -1)
	return Add(Mul(x, cos), Mul(rot, sin))
}

// Embed returns the L2-normalized sentence vectors [B, 768] of right-padded tokens [B, T] with int32 lengths [B].
func (m *Model) Embed(scope *model.Scope, tokens, lengths *Node) *Node {
	x, valid := m.TokenStates(scope, tokens, lengths)
	// The mean includes the prompt tokens, as sentence-transformers' Pooling does.
	m01 := ExpandAxes(ConvertDType(valid, dtypes.Float32), -1)
	pooled := Div(ReduceSum(Mul(x, m01), 1), ReduceSum(m01, 1))
	return Div(pooled, Sqrt(ReduceAndKeep(Square(pooled), ReduceSum, -1)))
}

// TokenStates returns the projected final token states [B, T, 768] that Embed averages and the real-token mask [B, T].
func (m *Model) TokenStates(scope *model.Scope, tokens, lengths *Node) (states, valid *Node) {
	g := tokens.Graph()
	c := m.cfg
	B, T := tokens.Shape().Dimensions[0], tokens.Shape().Dimensions[1]
	W := func(name string) *Node {
		v, ok := m.vars[name]
		if !ok {
			panic("embgemma: missing weight " + name)
		}
		return v.NodeValue(g)
	}
	eps := c.RMSNormEps
	H, P, L := c.HiddenSize, c.PerLayerInput, c.NumLayers

	pos := Iota(g, shapes.Make(dtypes.Int32, 1, T), 1)
	valid = LessThan(pos, Reshape(lengths, B, 1))

	x := Gather(W("embed_tokens.weight"), ExpandAxes(tokens, -1))
	x = MulScalar(x, math.Sqrt(float64(H)))

	// Per-layer inputs come from the embeddings alone (projection-only PLE).
	pl := MulScalar(linear(x, W("ple.per_layer_model_projection.weight")), 1/math.Sqrt(float64(H)))
	pl = rmsNorm(Reshape(pl, B, T, L, P), W("ple.per_layer_projection_norm.weight"), eps)

	// Sliding-window layers also need |query - key| <= sliding_window (transformers' sliding_window_bidirectional_overlay).
	keyMask := Reshape(valid, B, 1, 1, T)
	qi := Iota(g, shapes.Make(dtypes.Int32, T, T), 0)
	ki := Iota(g, shapes.Make(dtypes.Int32, T, T), 1)
	window := LessOrEqual(Abs(Sub(qi, ki)), Const(g, int32(c.SlidingWindow)))
	slidingMask := LogicalAnd(BroadcastToDims(keyMask, B, 1, T, T), Reshape(window, 1, 1, T, T))

	for i := range L {
		p := fmt.Sprintf("layers.%d.", i)
		hd, kv, theta := m.layer(i)
		nh := c.NumHeads

		h := rmsNorm(x, W(p+"input_layernorm.weight"), eps)
		q := Reshape(linear(h, W(p+"self_attn.q_proj.weight")), B, T, nh, hd)
		k := Reshape(linear(h, W(p+"self_attn.k_proj.weight")), B, T, kv, hd)
		v := Reshape(linear(h, W(p+"self_attn.v_proj.weight")), B, T, kv, hd)
		q = rope(rmsNorm(q, W(p+"self_attn.q_norm.weight"), eps), theta)
		k = rope(rmsNorm(k, W(p+"self_attn.k_norm.weight"), eps), theta)
		v = rmsNorm(v, nil, eps)
		// Grouped-query attention: query head j uses key/value head j / (nh / kv).
		q = Reshape(q, B, T, kv, nh/kv, hd)
		scores := Einsum("bqkgd,bskd->bkgqs", q, k) // scaling is 1: q and k are already normalized
		scores = Reshape(scores, B, nh, T, T)
		mask := keyMask
		if c.LayerTypes[i] == "sliding_attention" {
			mask = slidingMask
		}
		attn := MaskedSoftmax(scores, BroadcastToShape(mask, scores.Shape()), -1)
		attn = Reshape(attn, B, kv, nh/kv, T, T)
		o := Einsum("bkgqs,bskd->bqkgd", attn, v)
		o = linear(Reshape(o, B, T, nh*hd), W(p+"self_attn.o_proj.weight"))
		x = Add(x, rmsNorm(o, W(p+"post_attention_layernorm.weight"), eps))

		h = rmsNorm(x, W(p+"pre_feedforward_layernorm.weight"), eps)
		h = Mul(activation.GeluApproximate(linear(h, W(p+"mlp.gate_proj.weight"))), linear(h, W(p+"mlp.up_proj.weight")))
		h = linear(h, W(p+"mlp.down_proj.weight"))
		x = Add(x, rmsNorm(h, W(p+"post_feedforward_layernorm.weight"), eps))

		pli := Squeeze(Slice(pl, AxisRange(), AxisRange(), AxisElem(i), AxisRange()), 2)
		h = Mul(activation.GeluApproximate(linear(x, W(p+"ple_block.per_layer_input_gate.weight"))), pli)
		h = rmsNorm(linear(h, W(p+"ple_block.per_layer_projection.weight")), W(p+"ple_block.post_per_layer_input_norm.weight"), eps)
		x = Add(x, h)
		x = Mul(x, Reshape(W(p+"layer_scalar"))) // [1] -> scalar
	}
	x = rmsNorm(x, W("norm.weight"), eps)
	return linear(x, W("embedding_projection.weight")), valid
}
