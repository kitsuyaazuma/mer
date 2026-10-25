package systemone

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// Record is one line of a dataset: a wire request with its ID, where it comes from, and the right criterion of each
// labeled question (true or false for a noul, the zero-based level for a score).
type Record struct {
	ID     string `json:"id"`
	Source string `json:"source,omitempty"`
	WireRequest
	Labels map[string]string `json:"labels,omitempty"`
}

// ReadFile reads a dataset of one Record per line.
func ReadFile(path string) ([]Request, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []Request
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<26) // some states hold long tool lists
	for line := 1; sc.Scan(); line++ {
		var rec Record
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		r, err := rec.Request()
		if err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		out = append(out, r)
	}
	return out, sc.Err()
}

// Request parses the record and sets the gold criterion of its labeled questions.
func (rec Record) Request() (Request, error) {
	r, err := Parse(rec.WireRequest)
	if err != nil {
		return r, err
	}
	r.ID, r.Source = rec.ID, rec.Source
	for i := range r.Questions {
		q := &r.Questions[i]
		label, ok := rec.Labels[q.ID]
		if !ok {
			continue
		}
		for k, c := range q.Criteria {
			if c.Name == label {
				q.Gold = k
			}
		}
		if q.Gold < 0 {
			return r, fmt.Errorf("question %q: the label %q is not one of its criteria", q.ID, label)
		}
	}
	return r, nil
}

// Record writes r with the labels of its labeled questions.
func (r Request) Record() Record {
	rec := Record{ID: r.ID, Source: r.Source, WireRequest: r.Wire(), Labels: map[string]string{}}
	for _, q := range r.Questions {
		if q.Gold >= 0 {
			rec.Labels[q.ID] = q.Criteria[q.Gold].Name
		}
	}
	return rec
}

// WriteFile writes rs as a dataset of one Record per line.
func WriteFile(path string, rs []Request) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	for _, r := range rs {
		if err := enc.Encode(r.Record()); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
