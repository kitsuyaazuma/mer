package massive

import (
	"fmt"
	"hash/fnv"
	"math/rand"
	"slices"

	"github.com/gomlx/go-huggingface/tokenizers"
)

// LabelFn picks the label (an option ID of the schema) of an utterance.
type LabelFn func(Utterance) string

// Scenario and Intent are the two label granularities of MASSIVE.
var (
	Scenario LabelFn = func(u Utterance) string { return u.Scenario }
	Intent   LabelFn = func(u Utterance) string { return u.Intent }
)

// Splits holds the prompts of MASSIVE's official train, dev and test partitions.
type Splits struct {
	Train, Dev, Test []Prompt
	MaxLen           int // longest sequence in tokens over all splits
}

// BuildSplits keeps the utterances whose label is an option of the schema and tokenizes them.
// Options are shown in a random order fixed per utterance ID, so position is never a shortcut for the label.
func BuildSplits(tok tokenizers.Tokenizer, s Schema, utts []Utterance, label LabelFn, seed int64, limit, candidates int) (*Splits, error) {
	return BuildSplitsWith(tok, s, utts, label, SplitOptions{
		Seed:       seed,
		Limit:      limit,
		Candidates: candidates,
	})
}

// SplitOptions configures BuildSplitsWith.
type SplitOptions struct {
	Seed       int64
	Limit      int   // > 0: keep only the first Limit prompts of each split
	Candidates int   // > 0: show only that many options per prompt (the correct one + random distractors)
	Unseen     []int // options never shown in train/dev, not even as distractors; test draws from all options
}

// PickUnseen draws n options (canonical indices, sorted) that training will never see.
func PickUnseen(s Schema, n int, seed int64) []int {
	picked := rand.New(rand.NewSource(seed)).Perm(len(s.Options))[:n]
	slices.Sort(picked)
	return picked
}

// BuildSplitsWith is BuildSplits with all the options.
func BuildSplitsWith(tok tokenizers.Tokenizer, s Schema, utts []Utterance, label LabelFn, o SplitOptions) (*Splits, error) {
	seed, limit, candidates, unseen := o.Seed, o.Limit, o.Candidates, o.Unseen
	sp := &Splits{}
	for _, u := range utts {
		var out *[]Prompt
		switch u.Partition {
		case "train":
			out = &sp.Train
		case "dev":
			out = &sp.Dev
		case "test":
			out = &sp.Test
		default:
			continue
		}
		canon := s.Index(label(u))
		if canon < 0 {
			continue
		}
		hidden := u.Partition != "test" && len(unseen) > 0
		if hidden && slices.Contains(unseen, canon) {
			continue
		}
		if limit > 0 && len(*out) >= limit {
			continue
		}
		h := fnv.New64a()
		h.Write([]byte(u.Partition + "/" + u.ID))
		rng := rand.New(rand.NewSource(seed ^ int64(h.Sum64())))
		order := rng.Perm(len(s.Options))
		if hidden {
			order = slices.DeleteFunc(order, func(c int) bool { return slices.Contains(unseen, c) })
		}
		if candidates > 0 && candidates < len(order) {
			picked := []int{canon}
			for _, c := range order {
				if c != canon && len(picked) < candidates {
					picked = append(picked, c)
				}
			}
			rng.Shuffle(len(picked), func(i, j int) { picked[i], picked[j] = picked[j], picked[i] })
			order = picked
		}
		p, err := BuildPrompt(tok, s, order, u.Text, canon)
		if err != nil {
			return nil, fmt.Errorf("utterance %s/%s %q: %w", u.Partition, u.ID, u.Text, err)
		}
		p.SrcID = u.Partition + "/" + u.ID
		*out = append(*out, p)
		sp.MaxLen = max(sp.MaxLen, len(p.IDs))
	}
	return sp, nil
}
