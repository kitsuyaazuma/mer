package systemone

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

func TestParse(t *testing.T) {
	var w WireRequest
	body := `{"model": "mer", "state": {"user": "hi"}, "questions": {
		"b": {"type": "choice", "instructions": "Which?", "criteria": {"z": "last letter", "a": null}},
		"a": {"type": "noul", "instructions": ["is", "it"], "criteria": {"true": "it is"}},
		"c": {"type": "score", "instructions": "How much?", "criteria": ["none", "some", "a lot"]}}}`
	if err := json.Unmarshal([]byte(body), &w); err != nil {
		t.Fatal(err)
	}
	r, err := Parse(w)
	if err != nil {
		t.Fatal(err)
	}
	want := Request{State: `{"user":"hi"}`, Questions: []Question{
		{ID: "a", Instructions: `["is","it"]`, Type: Noul, Criteria: []Criterion{{"true", "it is"}, {"false", "no"}}, Gold: -1},
		{ID: "b", Instructions: "Which?", Type: Choice, Criteria: []Criterion{{"z", "last letter"}, {"a", ""}}, Gold: -1},
		{ID: "c", Instructions: "How much?", Type: Score, Criteria: []Criterion{{"0", "none"}, {"1", "some"}, {"2", "a lot"}}, Gold: -1},
	}}
	if !reflect.DeepEqual(r, want) {
		t.Errorf("Parse:\n got %+v\nwant %+v", r, want)
	}
}

func TestParseErrors(t *testing.T) {
	for name, body := range map[string]string{
		"blank state":     `{"state": " ", "questions": {"a": {"type": "noul", "instructions": "?"}}}`,
		"number state":    `{"state": 3, "questions": {"a": {"type": "noul", "instructions": "?"}}}`,
		"no questions":    `{"state": "s", "questions": {}}`,
		"one option":      `{"state": "s", "questions": {"a": {"type": "choice", "instructions": "?", "criteria": {"x": "y"}}}}`,
		"noul key":        `{"state": "s", "questions": {"a": {"type": "noul", "instructions": "?", "criteria": {"yes": "y"}}}}`,
		"unknown type":    `{"state": "s", "questions": {"a": {"type": "rank", "instructions": "?"}}}`,
		"images":          `{"state": "s", "images": ["AA=="], "questions": {"a": {"type": "noul", "instructions": "?"}}}`,
		"repeated option": `{"state": "s", "questions": {"a": {"type": "choice", "instructions": "?", "criteria": {"x": "1", "x": "2"}}}}`,
	} {
		var w WireRequest
		if err := json.Unmarshal([]byte(body), &w); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if _, err := Parse(w); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestAnswers(t *testing.T) {
	r := Request{Questions: []Question{
		{ID: "n", Type: Noul, Criteria: []Criterion{{"true", ""}, {"false", ""}}},
		{ID: "c", Type: Choice, Criteria: []Criterion{{"x", ""}, {"y", ""}}},
		{ID: "s", Type: Score, Criteria: []Criterion{{"0", "low"}, {"1", "mid"}, {"2", "high"}}},
	}}
	ln3 := float32(math.Log(3))
	got := Answers(r, [][]float32{{0, 0}, {0, 0}, {0, ln3, 0}})
	near := func(name string, got *float64, want float64) {
		t.Helper()
		if got == nil || math.Abs(*got-want) > 1e-6 {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
	near("noul", got["n"].Noul, 0.5)
	near("uniform confidence", got["c"].Confidence, 0)
	if got["c"].Choice != "x" {
		t.Errorf("a tie is the first option, got %q", got["c"].Choice)
	}
	near("score", got["s"].Score, 1) // p = 0.2, 0.6, 0.2
	if got["s"].Legend["2"] != "high" || math.Abs(got["s"].Probabilities["1"]-0.6) > 1e-6 {
		t.Errorf("score answer %+v", got["s"])
	}
}
