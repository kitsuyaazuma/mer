package toolbench

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// Action is what the assistant does next.
type Action int

// Next actions.
const (
	ActCall   Action = iota // call a tool
	ActAsk                  // ask the user for missing information
	ActCannot               // say it cannot be done with these tools
)

// Turn is one training state: the first user message of a dialog, its tools and what the assistant did.
type Turn struct {
	Query string
	Tools []Tool
	Act   Action
	Gold  []string // names of the tools called, empty unless Act is ActCall
}

var callRe = regexp.MustCompile(`(?:^\[|,\s*)([^\(\[\],=]+?)\(`)

// LoadToolACE reads data/toolace/data.json (Team-ACE/ToolACE, Apache-2.0): the first user turn of each dialog, when the reply calls tools.
func LoadToolACE(path string) ([]Turn, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var recs []struct {
		System        string `json:"system"`
		Conversations []struct {
			From, Value string
		} `json:"conversations"`
	}
	if err := json.Unmarshal(raw, &recs); err != nil {
		return nil, err
	}
	var out []Turn
	for _, r := range recs {
		i := strings.Index(r.System, "[{")
		if i < 0 {
			continue
		}
		var fns []json.RawMessage
		if err := json.NewDecoder(strings.NewReader(r.System[i:])).Decode(&fns); err != nil {
			continue
		}
		tools := make([]Tool, 0, len(fns))
		names := map[string]bool{}
		for _, f := range fns {
			t, err := parseTool(f)
			if err != nil {
				tools = nil
				break
			}
			tools = append(tools, t)
			names[t.Name] = true
		}
		if tools == nil {
			continue
		}
		var user, reply string
		for k, m := range r.Conversations {
			if m.From == "user" {
				user = m.Value
				if k+1 < len(r.Conversations) && r.Conversations[k+1].From == "assistant" {
					reply = strings.TrimSpace(r.Conversations[k+1].Value)
				}
				break
			}
		}
		if user == "" || !strings.HasPrefix(reply, "[") || !strings.HasSuffix(reply, ")]") {
			continue
		}
		t := Turn{Query: user, Tools: tools, Act: ActCall}
		ok := true
		for _, m := range callRe.FindAllStringSubmatch(reply, -1) {
			n := strings.TrimSpace(m[1])
			if !names[n] {
				ok = false
				break
			}
			t.Gold = append(t.Gold, n)
		}
		if !ok || len(t.Gold) == 0 {
			continue
		}
		out = append(out, t)
	}
	return out, nil
}

func asksBack(reply string) bool { return strings.Contains(reply, "?") }

// LoadWhen2CallTrain reads data/when2call/train/when2call_train_sft.jsonl (nvidia/When2Call, CC BY 4.0).
func LoadWhen2CallTrain(path string) ([]Turn, error) {
	var out []Turn
	err := eachLine(path, func(line []byte) error {
		var r struct {
			Tools    []string `json:"tools"`
			Messages []struct{ Role, Content string }
		}
		if err := json.Unmarshal(line, &r); err != nil {
			return err
		}
		if len(r.Messages) < 2 || r.Messages[0].Role != "user" || r.Messages[1].Role != "assistant" {
			return nil
		}
		t := Turn{Query: r.Messages[0].Content, Act: ActCannot}
		for _, s := range r.Tools {
			tool, err := parseTool(json.RawMessage(s))
			if err != nil {
				continue
			}
			t.Tools = append(t.Tools, tool)
		}
		if asksBack(r.Messages[1].Content) {
			t.Act = ActAsk
		}
		out = append(out, t)
		return nil
	})
	return out, err
}

// TestCase is one record of the When2Call test set, built from BFCL v2 live_simple and live_multiple.
type TestCase struct {
	UUID     string
	SourceID string
	Query    string
	Tools    []Tool
	Act      Action
	Target   string    // the tool to call (tool_call) or the one that lacks an argument (request_for_info)
	Answers  [4]string // direct, tool_call, request_for_info and cannot_answer, in the published order
	Gold     int       // index of the right one in Answers (never 0 by construction)
}

// LoadWhen2CallTest reads data/when2call/test/when2call_test_mcq.jsonl.
func LoadWhen2CallTest(path string) ([]TestCase, error) {
	var out []TestCase
	err := eachLine(path, func(line []byte) error {
		var r struct {
			UUID          string            `json:"uuid"`
			SourceID      string            `json:"source_id"`
			Question      string            `json:"question"`
			CorrectAnswer string            `json:"correct_answer"`
			Tools         []string          `json:"tools"`
			TargetTool    *string           `json:"target_tool"`
			Answers       map[string]string `json:"answers"`
		}
		if err := json.Unmarshal(line, &r); err != nil {
			return err
		}
		c := TestCase{UUID: r.UUID, SourceID: r.SourceID, Query: r.Question}
		switch r.CorrectAnswer {
		case "tool_call":
			c.Act, c.Gold = ActCall, 1
		case "request_for_info":
			c.Act, c.Gold = ActAsk, 2
		case "cannot_answer":
			c.Act, c.Gold = ActCannot, 3
		default:
			return fmt.Errorf("unknown correct_answer %q", r.CorrectAnswer)
		}
		for k, name := range []string{"direct", "tool_call", "request_for_info", "cannot_answer"} {
			a, ok := r.Answers[name]
			if !ok {
				return fmt.Errorf("%s: no %s answer", r.SourceID, name)
			}
			c.Answers[k] = a
		}
		for _, s := range r.Tools {
			t, err := parseTool(json.RawMessage(s))
			if err != nil {
				return fmt.Errorf("parsing the tools of %s: %w", r.SourceID, err)
			}
			c.Tools = append(c.Tools, t)
		}
		if r.TargetTool != nil {
			t, err := parseTool(json.RawMessage(*r.TargetTool))
			if err != nil {
				return fmt.Errorf("parsing the target tool of %s: %w", r.SourceID, err)
			}
			c.Target = t.Name
		}
		out = append(out, c)
		return nil
	})
	return out, err
}

// NormName folds a tool name for the overlap check between training and BFCL tools (lower case, letters and digits).
func NormName(n string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(n) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// PrefTurn is one record of When2Call's preference training set: the state, the chosen and the rejected reply.
type PrefTurn struct {
	Turn
	Chosen, Rejected string
}

// LoadWhen2CallPref reads data/when2call/train/when2call_train_pref.jsonl, labeling each turn from its chosen reply.
func LoadWhen2CallPref(path string) ([]PrefTurn, error) {
	var out []PrefTurn
	err := eachLine(path, func(line []byte) error {
		var r struct {
			Tools    []string `json:"tools"`
			Messages []struct{ Role, Content string }
			Chosen   struct{ Content string } `json:"chosen_response"`
			Rejected struct{ Content string } `json:"rejected_response"`
		}
		if err := json.Unmarshal(line, &r); err != nil {
			return err
		}
		if len(r.Messages) != 1 || r.Messages[0].Role != "user" {
			return nil
		}
		t := PrefTurn{Query: r.Messages[0].Content, Chosen: r.Chosen.Content, Rejected: r.Rejected.Content}
		for _, s := range r.Tools {
			if tool, err := parseTool(json.RawMessage(s)); err == nil {
				t.Tools = append(t.Tools, tool)
			}
		}
		if body, ok := strings.CutPrefix(strings.TrimSpace(t.Chosen), "<TOOLCALL>"); ok {
			var calls []struct {
				Name string `json:"name"`
			}
			if err := json.Unmarshal([]byte(strings.TrimSuffix(body, "</TOOLCALL>")), &calls); err != nil || len(calls) == 0 {
				return nil
			}
			for _, c := range calls {
				t.Gold = append(t.Gold, c.Name)
			}
			t.Act = ActCall
		} else if asksBack(t.Chosen) {
			t.Act = ActAsk
		} else {
			t.Act = ActCannot
		}
		out = append(out, t)
		return nil
	})
	return out, err
}
