package toolbench

import (
	"encoding/json"
	"math/rand"
	"regexp"
	"slices"
	"strings"

	"github.com/kitsuyaazuma/mer/internal/systemone"
)

// The wording of the Decision Index (decision_index/suite/build of apolinario/decision-index).
const (
	toolTask         = "BFCL tool-name selection"
	toolInstructions = "Given the complete user conversation and the published function schemas in state, mark each " +
		"candidate tool yes if it should be invoked to answer the request, or no otherwise. " +
		"This evaluates tool-name selection only; do not produce arguments or call ordering.\n\n"
	responseInstructions = "Which response should the assistant give to the user's question, given the available tools?"
)

var nonWord = regexp.MustCompile(`[^A-Za-z0-9_]+`)

// ToolQuestion asks whether one tool should be invoked, as a yes / no choice named tool_<name>.
func ToolQuestion(name string, called, known bool) systemone.Question {
	q := systemone.Question{ID: "tool_" + strings.Trim(nonWord.ReplaceAllString(name, "_"), "_"), Type: systemone.Choice,
		Instructions: toolInstructions + "\n\nCandidate tool: " + name,
		Criteria:     []systemone.Criterion{{Name: "yes", Desc: "Invoke this tool."}, {Name: "no", Desc: "Do not invoke this tool."}},
		Gold:         -1}
	if known {
		q.Gold = 1
		if called {
			q.Gold = 0
		}
	}
	return q
}

// ResponseQuestion asks which of the responses, named A, B, ..., the assistant should give.
func ResponseQuestion(responses []string, gold int) systemone.Question {
	q := systemone.Question{ID: "q", Type: systemone.Choice, Instructions: responseInstructions, Gold: gold}
	for i, r := range responses {
		q.Criteria = append(q.Criteria, systemone.Criterion{Name: string(rune('A' + i)), Desc: r})
	}
	return q
}

// toolRequest is a request about which of tools to invoke, with the state of the Index's BFCL cases.
func toolRequest(source, id string, conversation json.RawMessage, tools []Tool, gold []string, known bool) systemone.Request {
	fns := make([]json.RawMessage, len(tools))
	for i, t := range tools {
		fns[i] = t.Raw
	}
	r := systemone.Request{Source: source, ID: id, State: compact(struct {
		Conversation json.RawMessage   `json:"conversation"`
		Functions    []json.RawMessage `json:"functions"`
		Task         string            `json:"task"`
	}{conversation, fns, toolTask})}
	seen := map[string]bool{}
	for _, t := range tools {
		q := ToolQuestion(t.Name, slices.Contains(gold, t.Name), known)
		if !seen[q.ID] { // a question per name
			seen[q.ID] = true
			r.Questions = append(r.Questions, q)
		}
	}
	return r
}

// responseRequest is a request about which response to give, with the state of the Index's When2Call items.
func responseRequest(source, id, query string, tools []Tool, responses []string, gold int) systemone.Request {
	raw := make([]json.RawMessage, len(tools))
	for i, t := range tools {
		raw[i] = t.Raw
	}
	return systemone.Request{Source: source, ID: id, State: compact(struct {
		Tools    []json.RawMessage `json:"tools"`
		Question string            `json:"question"`
	}{raw, query}), Questions: []systemone.Question{ResponseQuestion(responses, gold)}}
}

// compact is v as compact JSON, as the Index sends a state.
func compact(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err) // raw messages read from JSON always marshal
	}
	return string(b)
}

// userTurn is a conversation of one user message, in BFCL's form.
func userTurn(query string) json.RawMessage {
	return json.RawMessage(compact([][]struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{{{"user", query}}}))
}

func limitTools(tools []Tool, gold []string, n int, rng *rand.Rand) []Tool {
	out := slices.Clone(tools)
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	if len(out) <= n {
		return out
	}
	slices.SortStableFunc(out, func(a, b Tool) int {
		ga, gb := slices.Contains(gold, a.Name), slices.Contains(gold, b.Name)
		switch {
		case ga && !gb:
			return -1
		case gb && !ga:
			return 1
		}
		return 0
	})
	out = out[:n]
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// FromBFCL builds the Index's request for a BFCL case: a yes / no question per tool, the tools in their published order.
func FromBFCL(c Case) systemone.Request {
	return toolRequest("bfcl:"+c.Category, c.ID, c.Conversation, c.Tools, c.Gold, true)
}

// FromToolACE builds a request like FromBFCL from a ToolACE call turn.
func FromToolACE(id string, t Turn, maxTools int, rng *rand.Rand) systemone.Request {
	return toolRequest("toolace", id, userTurn(t.Query), limitTools(t.Tools, t.Gold, maxTools, rng), t.Gold, true)
}

// FromWhen2CallSFT builds a request like FromBFCL from a When2Call refusal turn, where no tool is invoked. It builds
// none from a turn without tools, nor from a follow-up turn, whose labels are not known: the tool called once the user
// answers is not given.
func FromWhen2CallSFT(id string, t Turn, maxTools int, rng *rand.Rand) (systemone.Request, bool) {
	if t.Act != ActCannot || len(t.Tools) == 0 {
		return systemone.Request{}, false
	}
	return toolRequest("when2call-sft", id, userTurn(t.Query), limitTools(t.Tools, nil, maxTools, rng), nil, true), true
}

// FromWhen2CallPref builds a request like FromWhen2CallTest on the chosen and the rejected reply, in random order.
func FromWhen2CallPref(id string, t PrefTurn, maxTools int, rng *rand.Rand) systemone.Request {
	chosen, rejected := replyText(t.Chosen), replyText(t.Rejected)
	resp, gold := []string{chosen, rejected}, 0
	if rng.Intn(2) == 1 {
		resp, gold = []string{rejected, chosen}, 1
	}
	return responseRequest("when2call-pref", id, t.Query, limitTools(t.Tools, t.Gold, maxTools, rng), resp, gold)
}

// replyText writes a tool-calling reply as the When2Call test answers write it: each call as its JSON object, one per
// line, without the <TOOLCALL> tags of the training replies. Other replies are kept as they are.
func replyText(reply string) string {
	body, ok := strings.CutPrefix(strings.TrimSpace(reply), "<TOOLCALL>")
	if !ok {
		return reply
	}
	var calls []json.RawMessage
	if err := json.Unmarshal([]byte(strings.TrimSuffix(body, "</TOOLCALL>")), &calls); err != nil || len(calls) == 0 {
		return reply
	}
	lines := make([]string, len(calls))
	for i, c := range calls {
		lines[i] = string(c)
	}
	return strings.Join(lines, "\n")
}

// FromWhen2CallTest builds the Index's request for a When2Call test item: which of the four published answers to give.
func FromWhen2CallTest(c TestCase) systemone.Request {
	return responseRequest("when2call-test", c.UUID, c.Query, c.Tools, c.Answers[:], c.Gold)
}
