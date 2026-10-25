package main

import (
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/kitsuyaazuma/mer/internal/mer"
	"github.com/kitsuyaazuma/mer/internal/systemone"
)

// scores are the scores of one dataset. A case is a request whose questions are all labeled; it is right when every
// question is, as the Index scores BFCL (and, with one question per request, When2Call). With tau, the yes / no
// questions (two criteria, the first one yes) say yes when P(yes) >= tau.
type scores struct {
	nCase, okCase, okTau int
	tau                  float64
	nBin, yes, yesGold   int
	nTop1, okTop1        int // cases with two yes / no questions or more, one yes: is the most likely yes a right one?
	kinds                map[string]*accuracy
	bySource             map[string][2]int // the cases and the right ones, by where the requests come from
	byLen                [3]lenScores      // by the longest sequence of the request: up to 1,024 tokens, 2,048, longer
}

// lenScores are the cases, the right ones, the yes / no questions and the yes answers among requests of one length.
type lenScores struct{ n, ok, nBin, yes int }

// lenBucket is the bucket of byLen for a request whose longest sequence has n tokens.
func lenBucket(n int) int {
	switch {
	case n <= 1024:
		return 0
	case n <= 2048:
		return 1
	}
	return 2
}

// accuracy is that of the questions of one kind (their criteria, such as yes/no), overall and by the right criterion.
type accuracy struct {
	n, ok  int
	byGold map[string][2]int // the number asked and the number right
}

func (s scores) caseAcc() float64 { return ratio(s.okCase, s.nCase) }
func (s scores) tauAcc() float64 {
	if s.nBin == 0 {
		return math.NaN()
	}
	return ratio(s.okTau, s.nCase)
}
func (s scores) top1() float64   { return ratio(s.okTop1, s.nTop1) }
func (a *accuracy) acc() float64 { return ratio(a.ok, a.n) }

func (s scores) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "case %.4f (%d)", s.caseAcc(), s.nCase)
	if len(s.bySource) > 1 {
		sb.WriteString(" [")
		for i, src := range slices.Sorted(maps.Keys(s.bySource)) {
			if i > 0 {
				sb.WriteString(" ")
			}
			c := s.bySource[src]
			fmt.Fprintf(&sb, "%s %.4f (%d)", src, ratio(c[1], c[0]), c[0])
		}
		sb.WriteString("]")
	}
	if s.nBin > 0 {
		fmt.Fprintf(&sb, " case@tau %.4f (tau %.2f) top1 %.4f (%d) yes %.4f (gold %.4f)",
			s.tauAcc(), s.tau, s.top1(), s.nTop1, ratio(s.yes, s.nBin), ratio(s.yesGold, s.nBin))
	}
	if s.nBin > 0 && s.byLen[1].n+s.byLen[2].n > 0 {
		sb.WriteString(" | by length:")
		for b, name := range []string{"<=1024", "<=2048", ">2048"} {
			l := s.byLen[b]
			fmt.Fprintf(&sb, " %s case %.4f yes %.4f (%d)", name, ratio(l.ok, l.n), ratio(l.yes, l.nBin), l.n)
		}
	}
	for _, kind := range slices.Sorted(maps.Keys(s.kinds)) {
		a := s.kinds[kind]
		fmt.Fprintf(&sb, " | %s %.4f (", kind, a.acc())
		for i, name := range slices.Sorted(maps.Keys(a.byGold)) {
			if i > 0 {
				sb.WriteString(" ")
			}
			c := a.byGold[name]
			fmt.Fprintf(&sb, "%s %.3f", name, ratio(c[1], c[0]))
		}
		sb.WriteString(")")
	}
	return sb.String()
}

func ratio(a, b int) float64 {
	if b == 0 {
		return math.NaN()
	}
	return float64(a) / float64(b)
}

// signature names the kind of a question by its criteria, such as yes/no or A/B.
func signature(q systemone.Question) string {
	names := make([]string, len(q.Criteria))
	for k, c := range q.Criteria {
		names[k] = c.Name
	}
	return strings.Join(names, "/")
}

// evaluate scores the predicted logits perQ of every question of every request of rs; lens are the tokens of the
// longest sequence of each request.
func evaluate(rs []systemone.Request, perQ [][][]float32, lens []int, tau float64) scores {
	sc := scores{tau: tau, kinds: map[string]*accuracy{}, bySource: map[string][2]int{}}
	for i, r := range rs {
		labeled, okAll, okTau, nBin, nYes, nPred := true, true, true, 0, 0, 0
		top, topM, tie := -1, math.Inf(-1), false
		for qi, q := range r.Questions {
			lg := perQ[i][qi]
			if q.Gold < 0 || lg == nil {
				labeled = false
				continue
			}
			pred := argmax(lg)
			sc.add(q, pred)
			okAll = okAll && pred == q.Gold
			if len(lg) != 2 {
				okTau = okTau && pred == q.Gold
				continue
			}
			nBin++
			sc.nBin++
			p := systemone.Softmax(lg)[0]
			okTau = okTau && (p >= tau) == (q.Gold == 0)
			if pred == 0 {
				sc.yes++
				nPred++
			}
			if q.Gold == 0 {
				sc.yesGold++
				nYes++
			}
			// the margin orders the questions like P(yes) but does not saturate at 1; a tie is a miss
			switch m := float64(lg[0] - lg[1]); {
			case m > topM:
				top, topM, tie = qi, m, false
			case m == topM:
				tie = true
			}
		}
		if !labeled {
			continue
		}
		sc.nCase++
		c := sc.bySource[r.Source]
		c[0]++
		if okAll {
			sc.okCase++
			c[1]++
		}
		sc.bySource[r.Source] = c
		l := &sc.byLen[lenBucket(lens[i])]
		l.n, l.nBin, l.yes = l.n+1, l.nBin+nBin, l.yes+nPred
		if okAll {
			l.ok++
		}
		if okTau {
			sc.okTau++
		}
		if nYes > 0 && nBin >= 2 {
			sc.nTop1++
			if !tie && r.Questions[top].Gold == 0 {
				sc.okTop1++
			}
		}
	}
	return sc
}

func (s *scores) add(q systemone.Question, pred int) {
	kind := signature(q)
	a := s.kinds[kind]
	if a == nil {
		a = &accuracy{byGold: map[string][2]int{}}
		s.kinds[kind] = a
	}
	gold := q.Criteria[q.Gold].Name
	c := a.byGold[gold]
	c[0]++
	a.n++
	if pred == q.Gold {
		c[1]++
		a.ok++
	}
	a.byGold[gold] = c
}

// dump writes the probabilities of the criteria of every question predicted.
func dump(path string, rs []systemone.Request, perQ [][][]float32) {
	var sb strings.Builder
	sb.WriteString("id,question,gold,probs\n")
	for i, r := range rs {
		for qi, q := range r.Questions {
			lg := perQ[i][qi]
			if lg == nil {
				continue
			}
			var ps []string
			for _, p := range systemone.Softmax(lg) {
				ps = append(ps, fmt.Sprintf("%.4f", p))
			}
			fmt.Fprintf(&sb, "%q,%q,%d,%s\n", r.ID, q.ID, q.Gold, strings.Join(ps, ";"))
		}
	}
	must(os.MkdirAll(filepath.Dir(path), 0o755))
	must(os.WriteFile(path, []byte(sb.String()), 0o644))
}

func argmax(x []float32) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

// summary prints the mean and the standard deviation over the runs of the main scores of each design and dataset.
func summary(designs []mer.Design, names []string, results map[string][]map[string]scores) {
	fmt.Printf("\n================ SUMMARY (mean±sd over %d runs) ================\n", *flagRuns)
	for _, d := range designs {
		for _, name := range names {
			rs := results[d.Name]
			cols := []struct {
				label string
				get   func(s scores) float64
			}{
				{"case", scores.caseAcc}, {"case@tau", scores.tauAcc}, {"top1", scores.top1},
			}
			for _, kind := range slices.Sorted(maps.Keys(rs[0][name].kinds)) {
				cols = append(cols, struct {
					label string
					get   func(s scores) float64
				}{kind, func(s scores) float64 { return s.kinds[kind].acc() }})
			}
			var sb strings.Builder
			for _, c := range cols {
				var xs []float64
				for _, r := range rs {
					if v := c.get(r[name]); !math.IsNaN(v) {
						xs = append(xs, v)
					}
				}
				if len(xs) > 0 {
					m, sd := meanStd(xs)
					fmt.Fprintf(&sb, " | %s %.4f±%.4f", c.label, m, sd)
				}
			}
			fmt.Printf("%-20s %-5s%s\n", d.Name, name, sb.String())
		}
	}
}

func meanStd(xs []float64) (m, s float64) {
	for _, x := range xs {
		m += x
	}
	m /= float64(len(xs))
	for _, x := range xs {
		s += (x - m) * (x - m)
	}
	if len(xs) > 1 {
		s = math.Sqrt(s / float64(len(xs)-1))
	}
	return
}
