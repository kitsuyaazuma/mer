package systemone

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

// The wire format is that of POST /v1/systemone (https://docs.ollama.com/api/systemone).

// WireQuestion is one question on the wire. The criteria of a choice are an object of option keys and descriptions
// (null for a bare label), read in the order they are written; those of a noul are optional true and false
// descriptions; those of a score are a list of levels from the lowest (0) to the highest.
type WireQuestion struct {
	Type         QuestionType    `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

// WireRequest is a request on the wire. The state and the instructions are a nonblank string, or an object or array
// read as its JSON text. Mer reads no images, and keeps no model loaded or unloaded on keep_alive.
type WireRequest struct {
	Model     string                  `json:"model,omitempty"`
	State     json.RawMessage         `json:"state"`
	Images    []string                `json:"images,omitempty"`
	Questions map[string]WireQuestion `json:"questions"`
	KeepAlive json.RawMessage         `json:"keep_alive,omitempty"`
}

// Answer is the answer to one question: noul for a noul; choice, probabilities and confidence for a choice;
// score, legend, probabilities and confidence for a score.
type Answer struct {
	Type          QuestionType       `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

// Response is the reply to a WireRequest.
type Response struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   Usage             `json:"usage"`
}

// Usage counts the tokens read; the state is counted again for every question scored in its own sequence. Mer
// generates no tokens to score.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Limits of the wire format.
const (
	MaxQuestions = 64
	MaxChoices   = 255
	MaxLevels    = 10
)

// Parse turns a wire request into a Request with its questions in the sorted order of their names.
// A noul without criteria gets plain yes / no descriptions.
func Parse(w WireRequest) (Request, error) {
	var r Request
	if len(w.Images) > 0 {
		return r, errors.New("images are not supported")
	}
	var err error
	if r.State, err = content(w.State); err != nil {
		return r, fmt.Errorf("state: %w", err)
	}
	if n := len(w.Questions); n < 1 || n > MaxQuestions {
		return r, fmt.Errorf("questions: between 1 and %d are needed, not %d", MaxQuestions, n)
	}
	ids := make([]string, 0, len(w.Questions))
	for id := range w.Questions {
		if strings.TrimSpace(id) == "" {
			return r, errors.New("questions: a name is blank")
		}
		ids = append(ids, id)
	}
	slices.Sort(ids)
	for _, id := range ids {
		q, err := parseQuestion(id, w.Questions[id])
		if err != nil {
			return r, fmt.Errorf("question %q: %w", id, err)
		}
		r.Questions = append(r.Questions, q)
	}
	return r, nil
}

// content reads a nonblank string as the text itself, and an object or an array as its compact JSON.
func content(raw json.RawMessage) (string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return "", errors.New("missing")
	}
	switch raw[0] {
	case '"':
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", err
		}
		if strings.TrimSpace(s) == "" {
			return "", errors.New("blank")
		}
		return s, nil
	case '{', '[':
		var compact bytes.Buffer
		if err := json.Compact(&compact, raw); err != nil {
			return "", err
		}
		return compact.String(), nil
	}
	return "", errors.New("a nonblank string, an object or an array is needed")
}

func parseQuestion(id string, w WireQuestion) (Question, error) {
	q := Question{ID: id, Type: w.Type, Gold: -1}
	var err error
	if q.Instructions, err = content(w.Instructions); err != nil {
		return q, fmt.Errorf("instructions: %w", err)
	}
	switch w.Type {
	case Noul:
		q.Criteria = []Criterion{{"true", "yes"}, {"false", "no"}}
		if len(w.Criteria) > 0 {
			var given map[string]string
			if err := json.Unmarshal(w.Criteria, &given); err != nil {
				return q, fmt.Errorf("the criteria of a noul are true and false descriptions: %w", err)
			}
			for name, desc := range given {
				k := slices.IndexFunc(q.Criteria, func(c Criterion) bool { return c.Name == name })
				if k < 0 {
					return q, fmt.Errorf("the criteria of a noul are true and false, not %q", name)
				}
				q.Criteria[k].Desc = desc
			}
		}
	case Choice:
		if q.Criteria, err = orderedCriteria(w.Criteria); err != nil {
			return q, err
		}
		if n := len(q.Criteria); n < 2 || n > MaxChoices {
			return q, fmt.Errorf("between 2 and %d criteria are needed, not %d", MaxChoices, n)
		}
	case Score:
		var levels []string
		if err := json.Unmarshal(w.Criteria, &levels); err != nil {
			return q, fmt.Errorf("the criteria of a score are a list of levels: %w", err)
		}
		if n := len(levels); n < 2 || n > MaxLevels {
			return q, fmt.Errorf("between 2 and %d levels are needed, not %d", MaxLevels, n)
		}
		for i, l := range levels {
			q.Criteria = append(q.Criteria, Criterion{strconv.Itoa(i), l})
		}
	default:
		return q, fmt.Errorf("unknown type %q (choice | noul | score)", w.Type)
	}
	return q, nil
}

// orderedCriteria reads a JSON object of option keys and descriptions (or null), keeping the order they are written in.
func orderedCriteria(raw json.RawMessage) ([]Criterion, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	if t, err := dec.Token(); err != nil || t != json.Delim('{') {
		return nil, errors.New("the criteria of a choice are an object of option keys and descriptions")
	}
	var out []Criterion
	seen := map[string]bool{}
	for dec.More() {
		t, err := dec.Token()
		if err != nil {
			return nil, err
		}
		name := t.(string)
		if strings.TrimSpace(name) == "" || seen[name] {
			return nil, fmt.Errorf("option key %q is blank or repeated", name)
		}
		seen[name] = true
		var desc *string
		if err := dec.Decode(&desc); err != nil {
			return nil, fmt.Errorf("option %q: a description or null is needed", name)
		}
		c := Criterion{Name: name}
		if desc != nil {
			c.Desc = *desc
		}
		out = append(out, c)
	}
	return out, nil
}

// Wire writes r in the wire format, its state and instructions as JSON strings.
func (r Request) Wire() WireRequest {
	str := func(s string) json.RawMessage {
		b, _ := json.Marshal(s) // a string always marshals
		return b
	}
	w := WireRequest{State: str(r.State), Questions: map[string]WireQuestion{}}
	for _, q := range r.Questions {
		var b bytes.Buffer
		switch q.Type {
		case Score:
			levels := make([]string, len(q.Criteria))
			for k, c := range q.Criteria {
				levels[k] = c.Desc
			}
			raw, _ := json.Marshal(levels)
			b.Write(raw)
		default:
			b.WriteByte('{')
			for k, c := range q.Criteria {
				if k > 0 {
					b.WriteByte(',')
				}
				fmt.Fprintf(&b, "%s:%s", str(c.Name), str(c.Desc))
			}
			b.WriteByte('}')
		}
		w.Questions[q.ID] = WireQuestion{Type: q.Type, Instructions: str(q.Instructions), Criteria: b.Bytes()}
	}
	return w
}

// Answers turns the criterion logits of each question of r into the answers, keyed by question name.
func Answers(r Request, logits [][]float32) map[string]Answer {
	out := map[string]Answer{}
	for qi, q := range r.Questions {
		p := Softmax(logits[qi])
		a := Answer{Type: q.Type}
		if q.Type == Noul {
			a.Noul = &p[0]
			out[q.ID] = a
			continue
		}
		best, h := 0, 0.0
		a.Probabilities = map[string]float64{}
		for k, c := range q.Criteria {
			a.Probabilities[c.Name] = p[k]
			if p[k] > p[best] {
				best = k
			}
			if p[k] > 0 {
				h -= p[k] * math.Log(p[k])
			}
		}
		// how concentrated the probabilities are: 0 when uniform, near 1 when one criterion dominates
		conf := 1 - h/math.Log(float64(len(p)))
		a.Confidence = &conf
		if q.Type == Choice {
			a.Choice = q.Criteria[best].Name
		} else {
			// the expected level, so a score can fall between two levels
			score := 0.0
			a.Legend = map[string]string{}
			for k, c := range q.Criteria {
				score += p[k] * float64(k)
				a.Legend[c.Name] = c.Desc
			}
			a.Score = &score
		}
		out[q.ID] = a
	}
	return out
}

// Softmax turns logits into probabilities.
func Softmax(x []float32) []float64 {
	mx := math.Inf(-1)
	for _, v := range x {
		mx = math.Max(mx, float64(v))
	}
	out := make([]float64, len(x))
	z := 0.0
	for i, v := range x {
		out[i] = math.Exp(float64(v) - mx)
		z += out[i]
	}
	for i := range out {
		out[i] /= z
	}
	return out
}
