// Package tokspan tokenizes a text and finds the tokens that cover given byte ranges.
package tokspan

import (
	"fmt"
	"slices"

	"github.com/gomlx/go-huggingface/tokenizers"
)

// Encode tokenizes text and returns the token IDs and the [start,end) tokens of each byte range.
// It cuts the text at every range boundary and tokenizes the pieces one by one, so no token crosses a boundary.
func Encode(tok tokenizers.Tokenizer, text string, ranges [][2]int) (ids []int32, spans [][2]int32, err error) {
	cuts := []int{0, len(text)}
	for _, r := range ranges {
		cuts = append(cuts, r[0], r[1])
	}
	slices.Sort(cuts)
	cuts = slices.Compact(cuts)

	bos := len(tok.Encode("")) // Encode adds <bos> to every text; keep only the first one
	tokensBefore := map[int]int32{}
	for k := 0; k+1 < len(cuts); k++ {
		tokensBefore[cuts[k]] = int32(len(ids))
		piece := tok.Encode(text[cuts[k]:cuts[k+1]])
		if k > 0 {
			piece = piece[bos:]
		}
		for _, id := range piece {
			ids = append(ids, int32(id))
		}
	}
	tokensBefore[len(text)] = int32(len(ids))

	spans = make([][2]int32, len(ranges))
	for j, r := range ranges {
		spans[j] = [2]int32{tokensBefore[r[0]], tokensBefore[r[1]]}
		if spans[j][0] == spans[j][1] {
			return nil, nil, fmt.Errorf("byte range %d of %q: no token in range", j, text[r[0]:r[1]])
		}
	}
	return ids, spans, nil
}
