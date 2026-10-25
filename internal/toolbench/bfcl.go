// Package toolbench reads the BFCL, ToolACE and When2Call tool-calling data and turns it into System One requests in
// the form of the Jev Decision Index.
package toolbench

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// Tool is one function the request comes with.
type Tool struct {
	Name        string
	Description string
	Params      []Param
	Raw         json.RawMessage // the published JSON schema
}

// Param is one argument of a tool.
type Param struct {
	Name, Description string
	Required          bool
	Enum              []string
	Default           string // "" when there is none
}

// Case is one BFCL request with its tools.
type Case struct {
	ID           string
	Category     string
	Query        string          // the user messages, joined by newlines
	Conversation json.RawMessage // the published turns of messages
	Tools        []Tool
	Gold         []string // names of the tools called, possibly repeated; empty for the irrelevance categories
}

// Categories with a ground truth of tool calls, and the irrelevance ones (no call).
var (
	CallCategories       = []string{"simple", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"}
	IrrelevantCategories = []string{"irrelevance", "live_irrelevance"}
	// IndexCategories are the three categories of the Jev Decision Index (docs/suite.md of apolinario/decision-index).
	IndexCategories = []string{"simple", "live_simple", "live_multiple"}
)

// LoadBFCL reads one category of data/bfcl (BFCL v3 at gorilla commit 916260d, the Jev Decision Index revision).
func LoadBFCL(dir, category string) ([]Case, error) {
	gold := map[string][]string{}
	answers := filepath.Join(dir, "possible_answer_BFCL_v3_"+category+".json")
	if _, err := os.Stat(answers); err == nil { // the irrelevance categories have none
		err := eachLine(answers, func(line []byte) error {
			var a struct {
				ID          string           `json:"id"`
				GroundTruth []map[string]any `json:"ground_truth"`
			}
			if err := json.Unmarshal(line, &a); err != nil {
				return fmt.Errorf("parsing the answers of %s: %w", category, err)
			}
			for _, call := range a.GroundTruth {
				for name := range call {
					gold[a.ID] = append(gold[a.ID], name)
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	var cases []Case
	err := eachLine(filepath.Join(dir, "BFCL_v3_"+category+".json"), func(line []byte) error {
		var r struct {
			ID       string            `json:"id"`
			Question json.RawMessage   `json:"question"`
			Function []json.RawMessage `json:"function"`
		}
		if err := json.Unmarshal(line, &r); err != nil {
			return fmt.Errorf("parsing the %s cases: %w", category, err)
		}
		var turns [][]struct{ Role, Content string }
		if err := json.Unmarshal(r.Question, &turns); err != nil {
			return fmt.Errorf("parsing the messages of %s: %w", r.ID, err)
		}
		c := Case{ID: r.ID, Category: category, Conversation: r.Question, Gold: gold[r.ID]}
		var user []string
		for _, turn := range turns {
			for _, m := range turn {
				if m.Role == "user" {
					user = append(user, m.Content)
				}
			}
		}
		c.Query = strings.Join(user, "\n")
		for _, fn := range r.Function {
			t, err := parseTool(fn)
			if err != nil {
				return fmt.Errorf("parsing the tools of %s: %w", r.ID, err)
			}
			c.Tools = append(c.Tools, t)
		}
		cases = append(cases, c)
		return nil
	})
	return cases, err
}

// eachLine calls fn on every line of a JSONL file.
func eachLine(path string, fn func(line []byte) error) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<26) // some lines hold long tool lists
	for sc.Scan() {
		if err := fn(sc.Bytes()); err != nil {
			return err
		}
	}
	return sc.Err()
}

// parseTool reads one tool in the JSON Schema form BFCL, ToolACE and When2Call share; required parameters come first.
func parseTool(raw json.RawMessage) (Tool, error) {
	var fn struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Parameters  struct {
			Required   any                        `json:"required"`
			Properties map[string]json.RawMessage `json:"properties"`
		} `json:"parameters"`
	}
	if err := json.Unmarshal(raw, &fn); err != nil {
		return Tool{}, err
	}
	if fn.Name == "" {
		return Tool{}, errors.New("tool without a name")
	}
	t := Tool{Name: fn.Name, Description: fn.Description, Raw: raw}
	req := map[string]bool{}
	if l, ok := fn.Parameters.Required.([]any); ok {
		for _, p := range l {
			req[fmt.Sprint(p)] = true
		}
	}
	names := make([]string, 0, len(fn.Parameters.Properties))
	for n := range fn.Parameters.Properties {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool {
		if req[names[i]] != req[names[j]] {
			return req[names[i]]
		}
		return names[i] < names[j]
	})
	for _, n := range names {
		var p struct {
			Description string `json:"description"`
			Enum        []any  `json:"enum"`
			Default     any    `json:"default"`
		}
		_ = json.Unmarshal(fn.Parameters.Properties[n], &p)
		pp := Param{Name: n, Description: p.Description, Required: req[n]}
		for _, e := range p.Enum {
			pp.Enum = append(pp.Enum, fmt.Sprint(e))
		}
		if p.Default != nil {
			pp.Default = fmt.Sprint(p.Default)
		}
		t.Params = append(t.Params, pp)
	}
	return t, nil
}

// GoldInTools reports whether every gold name is one of the case's tools (one BFCL simple case is not).
func (c Case) GoldInTools() bool {
	names := map[string]bool{}
	for _, t := range c.Tools {
		names[t.Name] = true
	}
	for _, g := range c.Gold {
		if !names[g] {
			return false
		}
	}
	return true
}

// Text renders a tool as one line with its description and parameters, cut to maxChars bytes.
func (t Tool) Text(maxChars int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %s", t.Name, strings.TrimSpace(t.Description))
	if len(t.Params) > 0 {
		b.WriteString(" Parameters:")
		for i, p := range t.Params {
			if i > 0 {
				b.WriteString(";")
			}
			fmt.Fprintf(&b, " %s", p.Name)
			if p.Required {
				b.WriteString(" (required)")
			}
			if d := strings.TrimSpace(p.Description); d != "" {
				fmt.Fprintf(&b, ": %s", d)
			}
			if len(p.Enum) > 0 {
				fmt.Fprintf(&b, " [one of %s]", strings.Join(p.Enum, ", "))
			}
			if p.Default != "" {
				fmt.Fprintf(&b, " [default %s]", p.Default)
			}
		}
	}
	s := b.String()
	if maxChars > 0 && len(s) > maxChars {
		s = s[:maxChars]
		for !utf8.ValidString(s) {
			s = s[:len(s)-1]
		}
	}
	return s
}
