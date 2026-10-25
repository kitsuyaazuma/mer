// Package systemone is the request a Decision Model answers, as Ollama's System One API defines it: a state and
// questions about it with their criteria, in memory, on the wire (POST /v1/systemone) and as labeled JSONL datasets.
package systemone

import (
	"fmt"
	"strings"
)

// Criterion is one possible answer of a question: its name and what it means.
type Criterion struct{ Name, Desc string }

// QuestionType is the kind of a question: a yes/no noul, a choice among criteria, or a score on ordered levels.
type QuestionType string

// Question types.
const (
	Noul   QuestionType = "noul"
	Choice QuestionType = "choice"
	Score  QuestionType = "score"
)

// Question is one question of a request; the criteria of a noul are true and false, in this order.
type Question struct {
	ID, Instructions string
	Type             QuestionType
	Criteria         []Criterion
	Gold             int // index of the right criterion, -1 when unlabeled
}

// Request is a state given as text and the questions asked about it.
type Request struct {
	ID, Source string // Source names the data the request comes from, for the statistics
	State      string
	Questions  []Question
}

// Rendered is one question of a request written as one text, with the byte ranges of the state and of each criterion
// line.
type Rendered struct {
	Text       string
	StateRange [2]int
	OptRanges  [][2]int
}

// Render writes the state and then question qi with its criteria, one per line.
func (r Request) Render(qi int) Rendered {
	var sb strings.Builder
	sb.WriteString("STATE:\n")
	lo := sb.Len()
	sb.WriteString(r.State)
	out := Rendered{StateRange: [2]int{lo, sb.Len()}}
	q := r.Questions[qi]
	fmt.Fprintf(&sb, "\nQUESTION:\n%s\nOPTIONS:", q.Instructions)
	for _, c := range q.Criteria {
		sb.WriteString("\n")
		lo := sb.Len()
		sb.WriteString(c.Name)
		if c.Desc != "" {
			fmt.Fprintf(&sb, ": %s", c.Desc)
		}
		out.OptRanges = append(out.OptRanges, [2]int{lo, sb.Len()})
	}
	out.Text = sb.String()
	return out
}
