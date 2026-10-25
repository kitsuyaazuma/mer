// Package mer turns System One requests into the sequences Mer reads, one per question with the whole state, scores
// their criteria with a head, and saves and loads a trained Mer.
package mer

import (
	"runtime"
	"sync"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/systemone"
)

// Example is one question of a request written as one sequence: its tokens and where its state and its criteria are.
type Example struct {
	Req       *systemone.Request
	Parent, Q int // the index of the request among those read, and of the question in the request
	IDs       []int32
	StateSpan [2]int32
	Spans     [][2]int32 // the tokens of each criterion
}

// Question is the question of e.
func (e *Example) Question() systemone.Question { return e.Req.Questions[e.Q] }

// K is the number of criteria of e.
func (e *Example) K() int { return len(e.Spans) }

// NewExamples writes every question of rs as its own sequence, tokenizing on all the CPUs. Unlabeled questions are
// left out unless keepUnlabeled.
func NewExamples(bb *backbone.Backbone, rs []systemone.Request, keepUnlabeled bool) ([]Example, error) {
	var out []Example
	for i := range rs {
		for qi, q := range rs[i].Questions {
			if q.Gold >= 0 || keepUnlabeled {
				out = append(out, Example{Req: &rs[i], Parent: i, Q: qi})
			}
		}
	}
	tok := newPieceTok(bb.Tok)
	errs := make([]error, len(out))
	next := make(chan int)
	var wg sync.WaitGroup
	for range runtime.NumCPU() {
		wg.Go(func() {
			for i := range next {
				e := &out[i]
				rd := e.Req.Render(e.Q)
				ids, spans, err := encode(bb, tok, rd.Text, append([][2]int{rd.StateRange}, rd.OptRanges...))
				if err != nil {
					errs[i] = err
					continue
				}
				e.IDs, e.StateSpan, e.Spans = ids, spans[0], spans[1:]
			}
		})
	}
	for i := range out {
		next <- i
	}
	close(next)
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// PerQuestion puts the criterion logits lg of the sequences exs in their places among the questions of rs, the requests
// they were written from; questions not written stay nil.
func PerQuestion(rs []systemone.Request, exs []Example, lg [][]float32) [][][]float32 {
	out := make([][][]float32, len(rs))
	for i := range out {
		out[i] = make([][]float32, len(rs[i].Questions))
	}
	for d := range exs {
		e := &exs[d]
		out[e.Parent][e.Q] = lg[d][:e.K()]
	}
	return out
}
