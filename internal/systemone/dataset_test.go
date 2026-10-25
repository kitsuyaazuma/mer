package systemone

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestRecordRoundTrip(t *testing.T) {
	r := Request{ID: "r1", Source: "test", State: "request: hi", Questions: []Question{
		{ID: "call:f", Instructions: "Call f?", Type: Noul, Criteria: []Criterion{{"true", "yes, now"}, {"false", "not now"}}, Gold: 0},
		{ID: "next", Instructions: "What next?", Type: Choice, Criteria: []Criterion{{"b", "second"}, {"a", "first"}}, Gold: 1},
		{ID: "size", Instructions: "How big?", Type: Score, Criteria: []Criterion{{"0", "small"}, {"1", "big"}}, Gold: -1},
	}}
	b, err := json.Marshal(r.Record())
	if err != nil {
		t.Fatal(err)
	}
	var rec Record
	if err := json.Unmarshal(b, &rec); err != nil {
		t.Fatal(err)
	}
	got, err := rec.Request()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, r) {
		t.Errorf("round trip:\n got %+v\nwant %+v", got, r)
	}
}
