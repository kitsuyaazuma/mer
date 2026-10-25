package mer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/systemone"
)

// Meta is what Save writes next to the weights, so that Load rebuilds the same Mer.
type Meta struct {
	Design   string `json:"design"`
	Backbone string `json:"backbone"` // a short name or a Hugging Face repository ID
	DType    string `json:"dtype"`
	LoRA     bool   `json:"lora"`
}

const metaFile = "mer.json"

// Save writes the trained weights of store (the head, and with LoRA its adapters) and m into dir, which must be empty
// or not exist yet. The frozen backbone weights and the optimizer state are left out.
func Save(dir string, store *model.Store, m Meta) error {
	if entries, err := os.ReadDir(dir); err == nil && len(entries) > 0 {
		return fmt.Errorf("%s is not empty", dir)
	}
	var skip []*model.Variable
	for v := range store.IterVariables() {
		if !v.Trainable || backbone.IsOptimizerState(v.Path()) {
			skip = append(skip, v)
		}
	}
	h, err := checkpoint.Build(store).Dir(dir).ExcludeVars(skip...).Done()
	if err != nil {
		return err
	}
	if err := h.Save(); err != nil {
		return err
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, metaFile), append(b, '\n'), 0o644)
}

// Model is a trained Mer, loaded to answer requests.
type Model struct {
	Meta   Meta
	design Design
	bb     *backbone.Backbone
	exec   *model.Exec
	hidden *model.Exec // the backbone when the head reads its hidden states apart: frozen, or with LoRA and the Cross-field
}

// Load rebuilds the Mer that Save wrote into dir.
func Load(backend compute.Backend, dir string) (*Model, error) {
	b, err := os.ReadFile(filepath.Join(dir, metaFile))
	if err != nil {
		return nil, err
	}
	m := &Model{}
	if err := json.Unmarshal(b, &m.Meta); err != nil {
		return nil, err
	}
	if m.design, err = ParseDesign(m.Meta.Design); err != nil {
		return nil, err
	}
	repoID, err := backbone.RepoID(m.Meta.Backbone)
	if err != nil {
		return nil, err
	}
	dtype, err := backbone.ParseDType(m.Meta.DType)
	if err != nil {
		return nil, err
	}
	if m.bb, err = backbone.LoadDType(backend, repoID, dtype); err != nil {
		return nil, err
	}
	if m.Meta.LoRA {
		SetLoRA(m.bb)
		if _, err := checkpoint.Load(m.bb.Store).Dir(dir).Done(); err != nil {
			return nil, err
		}
		if m.design.CrossField {
			m.exec, m.hidden = m.design.HeadExec(backend, m.bb.Store), HiddenExec(backend, m.bb)
		} else {
			m.exec = m.design.LoRAExec(backend, m.bb)
		}
		return m, nil
	}
	store := model.NewStore()
	if _, err := checkpoint.Load(store).Dir(dir).Done(); err != nil {
		return nil, err
	}
	m.exec, m.hidden = m.design.HeadExec(backend, store), HiddenExec(backend, m.bb)
	return m, nil
}

// Answer returns the criterion logits of each question of r, in the order of r.Questions, and the number of tokens
// read.
func (m *Model) Answer(r systemone.Request) (logits [][]float32, tokens int, err error) {
	rs := []systemone.Request{r}
	exs, err := NewExamples(m.bb, rs, true)
	if err != nil {
		return nil, 0, err
	}
	for _, e := range exs {
		tokens += len(e.IDs)
	}
	inputs := func(idx []int, B, T, K int) ([]*tensors.Tensor, error) {
		return m.design.LoRAInputs(exs, m.bb.PadID, idx, B, T, K), nil
	}
	if m.hidden != nil {
		c, err := HiddenStates(m.hidden, m.bb, exs, "", "")
		if err != nil {
			return nil, 0, err
		}
		inputs = func(idx []int, B, T, K int) ([]*tensors.Tensor, error) {
			return m.design.FrozenInputs(exs, c, idx, B, T, K)
		}
	}
	lg, err := Predict(m.exec, exs, m.design.CrossField, inputs)
	if err != nil {
		return nil, 0, err
	}
	return PerQuestion(rs, exs, lg)[0], tokens, nil
}
