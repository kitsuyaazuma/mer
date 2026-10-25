// Package backbone loads a pre-trained decoder-only Transformer (Gemma 3 270M by default) and exposes its final hidden
// states.
package backbone

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

// DefaultRepoID is the ID of an un-gated mirror of google/gemma-3-270m with identical files.
const DefaultRepoID = "unsloth/gemma-3-270m"

// Default is the name of the backbone the article uses.
const Default = "gemma3-270m"

// Names are the backbones that can be picked by a short name, with their Hugging Face repositories.
var Names = map[string]string{
	Default:      DefaultRepoID,
	"gemma4-e4b": "google/gemma-4-E4B",
}

// RepoID is the repository of a backbone given by a short name or as a Hugging Face repository ID ("owner/name").
func RepoID(name string) (string, error) {
	if id, ok := Names[name]; ok {
		return id, nil
	}
	if strings.Contains(name, "/") {
		return name, nil
	}
	return "", fmt.Errorf("unknown backbone %q: a short name (gemma3-270m, gemma4-e4b) or a Hugging Face repository ID", name)
}

// Backbone is a pre-trained decoder-only Transformer, without its LM head.
type Backbone struct {
	RepoID    string
	HF        *transformer.Model
	Tok       tokenizers.Tokenizer
	Store     *model.Store
	HiddenDim int
	PadID     int32        // the padding token, put after the end of a sequence
	DType     dtypes.DType // of the weights
}

// Load downloads the model and loads it in float32 (faster than bfloat16 on CPU), frozen.
func Load(backend compute.Backend, repoID string) (*Backbone, error) {
	return LoadDType(backend, repoID, dtypes.Float32)
}

// LoadDType is Load with the weights in dtype: float32, or bfloat16 to halve the memory of a bigger model on a GPU.
func LoadDType(backend compute.Backend, repoID string, dtype dtypes.DType) (*Backbone, error) {
	if dtype != dtypes.Float32 {
		// cuDNN's fused attention takes head_dim <= 128 (Gemma has 256 and 512), but go-xla does not check it in
		// bfloat16: remove the capability (the map is the backend's own) so attention falls back to the plain version.
		ops := backend.Capabilities().Operations
		delete(ops, compute.OpTypeFusedScaledDotProductAttention)
		delete(ops, compute.OpTypeFusedScaledDotProductAttentionVJP)
	}
	repo := hub.New(repoID)
	if err := repo.DownloadInfo(false); err != nil {
		return nil, err
	}
	// The configs, the tokenizer and the weights, whether in one safetensors file or in shards.
	var files []string
	for name, err := range repo.IterFileNames() {
		if err != nil {
			return nil, err
		}
		if strings.HasSuffix(name, ".json") || strings.HasSuffix(name, ".safetensors") || name == "tokenizer.model" {
			files = append(files, name)
		}
	}
	if _, err := repo.DownloadFiles(files...); err != nil {
		return nil, err
	}
	hf, err := transformer.LoadModel(repo)
	if err != nil {
		return nil, err
	}

	tok, err := hf.GetTokenizer()
	if err != nil {
		return nil, err
	}
	store := model.NewStore()
	if err := hf.LoadStore(backend, store); err != nil {
		return nil, err
	}
	// LoadStore adds a copy of the tied embeddings as LM head weights, which only generation needs (~700MB for Gemma 3 270M)
	if store.GetVariable("/output/dense/weights") != nil {
		if err := store.DeleteVariable("/output/dense/weights"); err != nil {
			return nil, err
		}
	}
	if err := convertTo(backend, store, dtype); err != nil {
		return nil, err
	}
	name := strings.ToLower(dtype.String())
	hf.Config.TorchDtype, hf.Config.DType = name, name // so the graph is built in dtype

	b := &Backbone{RepoID: repoID, HF: hf, Tok: tok, Store: store, HiddenDim: hf.Config.HiddenSize, PadID: int32(hf.Config.PadTokenID), DType: dtype}
	b.Freeze()
	return b, nil
}

func convertTo(backend compute.Backend, store *model.Store, dtype dtypes.DType) error {
	for v := range store.IterVariables() {
		val, err := v.Value()
		if err != nil {
			return err
		}
		if val.DType() == dtype {
			continue
		}
		converted, err := CallOnce(backend, func(x *Node) *Node { return ConvertDType(x, dtype) }, val)
		if err != nil {
			return fmt.Errorf("converting %s: %w", v.Path(), err)
		}
		if err := v.SetValue(converted); err != nil {
			return err
		}
	}
	return nil
}

// Freeze marks every variable currently in the store as non-trainable.
func (b *Backbone) Freeze() {
	for v := range b.Store.IterVariables() {
		v.SetTrainable(false)
	}
}

// HiddenStates returns the float32 final hidden states [batch, seqLen, hiddenDim] for tokens [batch, seqLen].
// Tokens may be right-padded (with any token) because the causal attention keeps padding from influencing real tokens.
func (b *Backbone) HiddenStates(scope *model.Scope, tokens *Node) *Node {
	last, _ := b.HF.AllLayers(scope, tokens, nil)
	return ConvertDType(last, dtypes.Float32)
}

// IsOptimizerState tells whether a variable belongs to the optimizer (Adam moments, step counters).
func IsOptimizerState(path string) bool {
	return strings.HasPrefix(path, "/"+optimizer.AdamDefaultScope+"/") || strings.HasPrefix(path, "/"+optimizer.Scope+"/") || strings.HasSuffix(path, optimizer.GlobalStepVariableName)
}

// ParamCounts returns the number of trainable and frozen parameters in the store, excluding optimizer state.
func ParamCounts(store *model.Store) (trainable, frozen int) {
	for v := range store.IterVariables() {
		if IsOptimizerState(v.Path()) {
			continue
		}
		n := v.Shape().Size()
		if v.Trainable {
			trainable += n
		} else {
			frozen += n
		}
	}
	return
}

// Describe summarizes the loaded model.
func (b *Backbone) Describe() string {
	c := b.HF.Config
	return fmt.Sprintf("%s: %d layers, hidden=%d, heads=%d (kv=%d, head_dim=%d), ffn=%d, vocab=%d",
		b.HF.Repo.ID, c.NumHiddenLayers, c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, c.IntermediateSize, c.VocabSize)
}

// ParseDType reads the dtype of the weights: float32 or bfloat16.
func ParseDType(name string) (dtypes.DType, error) {
	switch name {
	case "float32":
		return dtypes.Float32, nil
	case "bfloat16":
		return dtypes.BFloat16, nil
	}
	return dtypes.InvalidDType, fmt.Errorf("unknown dtype %q (float32 | bfloat16)", name)
}
