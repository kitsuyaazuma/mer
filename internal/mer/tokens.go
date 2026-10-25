package mer

import (
	"encoding/gob"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/gomlx/go-huggingface/tokenizers"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/tokspan"
)

// Tokenizing all the requests takes minutes, and depends only on the tokenizer, the text and the ranges, so the
// tokens can be kept between runs (LoadTokenCache).

type tokEntry struct {
	IDs   []int32
	Spans [][2]int32
}

var tokCache = struct {
	mu    sync.Mutex // encode runs on several goroutines
	m     map[uint64]tokEntry
	path  string
	dirty bool
}{m: map[uint64]tokEntry{}}

func tokKey(repo, text string, ranges [][2]int) uint64 {
	h := fnv.New64a()
	fmt.Fprintf(h, "%s\x00%s", repo, text)
	for _, r := range ranges {
		fmt.Fprintf(h, "\x00%d,%d", r[0], r[1])
	}
	return h.Sum64()
}

// LoadTokenCache reads the tokenized texts kept in dir, and keeps the new ones there on SaveTokenCache.
func LoadTokenCache(dir string) {
	tokCache.mu.Lock()
	defer tokCache.mu.Unlock()
	tokCache.path = filepath.Join(dir, "mer-tokens.gob")
	f, err := os.Open(tokCache.path)
	if err != nil {
		return
	}
	defer f.Close()
	m := map[uint64]tokEntry{}
	if err := gob.NewDecoder(f).Decode(&m); err != nil {
		fmt.Printf("  token cache %s unreadable (%v): tokenizing again\n", tokCache.path, err)
		return
	}
	tokCache.m = m
	fmt.Printf("  token cache: %d tokenized texts loaded from %s\n", len(m), tokCache.path)
}

// SaveTokenCache writes the tokenized texts to the directory of LoadTokenCache, if any text is new.
func SaveTokenCache() error {
	tokCache.mu.Lock()
	defer tokCache.mu.Unlock()
	if tokCache.path == "" || !tokCache.dirty {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(tokCache.path), 0o755); err != nil {
		return err
	}
	tmp := tokCache.path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := gob.NewEncoder(f).Encode(tokCache.m); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	tokCache.dirty = false
	return os.Rename(tmp, tokCache.path)
}

// encode tokenizes text with the token ranges of the byte ranges, through the cache when one is loaded.
func encode(bb *backbone.Backbone, tok *pieceTok, text string, ranges [][2]int) ([]int32, [][2]int32, error) {
	tokCache.mu.Lock()
	keep := tokCache.path != ""
	tokCache.mu.Unlock()
	if !keep {
		return tokspan.Encode(tok, text, ranges)
	}
	k := tokKey(bb.RepoID, text, ranges)
	tokCache.mu.Lock()
	e, ok := tokCache.m[k]
	tokCache.mu.Unlock()
	if ok && len(e.Spans) == len(ranges) {
		return e.IDs, e.Spans, nil
	}
	ids, spans, err := tokspan.Encode(tok, text, ranges)
	if err != nil {
		return nil, nil, err
	}
	tokCache.mu.Lock()
	tokCache.m[k] = tokEntry{IDs: ids, Spans: spans}
	tokCache.dirty = true
	tokCache.mu.Unlock()
	return ids, spans, nil
}

// pieceTok remembers the tokens of every piece tokspan.Encode cuts a text into. Every question of a request is
// written after the same state, so the state is tokenized once per request instead of once per question.
type pieceTok struct {
	tokenizers.Tokenizer
	mu sync.Mutex
	m  map[string][]int
}

func newPieceTok(tok tokenizers.Tokenizer) *pieceTok {
	return &pieceTok{Tokenizer: tok, m: map[string][]int{}}
}

func (t *pieceTok) Encode(text string) []int {
	t.mu.Lock()
	ids, ok := t.m[text]
	t.mu.Unlock()
	if ok {
		return ids
	}
	ids = t.Tokenizer.Encode(text)
	t.mu.Lock()
	t.m[strings.Clone(text)] = ids // a copy, so the key does not keep the whole text alive
	t.mu.Unlock()
	return ids
}
