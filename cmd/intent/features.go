package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"os"
	"sort"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/go-huggingface/tokenizers"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"

	"github.com/kitsuyaazuma/mer/internal/embgemma"
	"github.com/kitsuyaazuma/mer/internal/tokspan"
)

type item struct {
	text   string // the whole sequence given to the encoder
	lo, hi int    // the byte range of text that is averaged
}

type featureBank struct {
	H    int            // the length of a row (the encoder's hidden size)
	rows [][]float32    // the mean vector of each text, in the order add saw them
	idx  map[string]int // text -> its row
}

func newFeatureBank(H int) *featureBank { return &featureBank{H: H, idx: map[string]int{}} }

func (f *featureBank) add(key string, it item, pending *[]item) int {
	if i, ok := f.idx[key]; ok {
		return i
	}
	i := len(f.idx)
	f.idx[key] = i
	*pending = append(*pending, it)
	return i
}

func encode(exec *model.Exec, tok tokenizers.Tokenizer, items []item, H, bs int, name string) [][]float32 {
	type enc struct {
		ids    []int32
		lo, hi int32
	}
	// Find the tokens of each item's byte range.
	encs := make([]enc, len(items))
	for i, it := range items {
		ids, spans := must2(tokspan.Encode(tok, it.text, [][2]int{{it.lo, it.hi}}))
		encs[i] = enc{ids, spans[0][0], spans[0][1]}
	}
	// Longest first, so that a batch holds sequences of about the same length.
	order := make([]int, len(items))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(encs[order[a]].ids) > len(encs[order[b]].ids) })
	out := make([][]float32, len(items))
	t0 := time.Now()
	tokens := 0
	for start := 0; start < len(order); start += bs {
		// Right-pad the chunk into one [bs, T] batch, T being its longest sequence rounded up to a multiple of 16.
		chunk := order[start:min(start+bs, len(order))]
		T := (len(encs[chunk[0]].ids) + 15) / 16 * 16
		// toks is the [bs, T] batch flattened, all <pad> to start. A short last chunk leaves its missing rows
		// as <pad>, so the shape stays [bs, T] and the graph compiled for this T is reused.
		toks := make([]int32, bs*T)
		for r, i := range chunk {
			copy(toks[r*T:], encs[i].ids)
			tokens += len(encs[i].ids)
		}
		// Run the frozen encoder: hidden states [bs, T, H].
		hidden := exec.MustCall1(tensors.FromFlatDataAndDimensions(toks, bs, T))
		flat := tensors.MustCopyFlatData[float32](hidden)
		_ = hidden.FinalizeAll()
		// Average the hidden states over the tokens of each item's range; out keeps the items' order.
		for r, i := range chunk {
			v := make([]float32, H)
			for t := encs[i].lo; t < encs[i].hi; t++ {
				row := flat[(r*T+int(t))*H : (r*T+int(t)+1)*H]
				for h := range v {
					v[h] += row[h]
				}
			}
			n := float32(encs[i].hi - encs[i].lo)
			for h := range v {
				v[h] /= n
			}
			out[i] = v
		}
	}
	fmt.Printf("  features %-9s %6d sequences (%d tokens) in %v\n", name, len(items), tokens, time.Since(t0).Round(time.Second))
	return out
}

func (f *featureBank) fill(exec *model.Exec, tok tokenizers.Tokenizer, pending []item, name string) {
	h := fnv.New64a()
	for _, it := range pending {
		fmt.Fprintf(h, "%s\x00%d\x00%d\x00", it.text, it.lo, it.hi)
	}
	path := ""
	if *flagCacheDir != "" {
		must(os.MkdirAll(*flagCacheDir, 0o755))
		path = fmt.Sprintf("%s/feat-%s-%x.f32", *flagCacheDir, name, h.Sum64())
		if rows, err := loadRows(path, len(pending), f.H); err == nil {
			f.rows = append(f.rows, rows...)
			fmt.Printf("  features %-9s %6d loaded from %s\n", name, len(rows), path)
			return
		}
	}
	rows := encode(exec, tok, pending, f.H, encBatch, name)
	if path != "" {
		must(saveRows(path, rows))
	}
	f.rows = append(f.rows, rows...)
}

func (f *featureBank) fillEmbeddings(backend compute.Backend, em *embgemma.Model, pending []item, prompt, name string) {
	texts := make([]string, len(pending))
	h := fnv.New64a()
	for i, it := range pending {
		texts[i] = prompt + it.text[it.lo:it.hi]
		fmt.Fprintf(h, "%s\x00", texts[i])
	}
	path := ""
	if *flagCacheDir != "" {
		must(os.MkdirAll(*flagCacheDir, 0o755))
		path = fmt.Sprintf("%s/feat-%s-%x.f32", *flagCacheDir, name, h.Sum64())
		if rows, err := loadRows(path, len(texts), f.H); err == nil {
			f.rows = append(f.rows, rows...)
			fmt.Printf("  features %-9s %6d loaded from %s\n", name, len(rows), path)
			return
		}
	}
	exec := model.MustNewExec(backend, em.Store, func(scope *model.Scope, tokens, lengths *Node) *Node { return em.Embed(scope, tokens, lengths) })
	ids := make([][]int32, len(texts))
	order := make([]int, len(texts))
	for i, t := range texts {
		ids[i], order[i] = em.Encode(t), i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(ids[order[a]]) > len(ids[order[b]]) })
	rows := make([][]float32, len(texts))
	t0 := time.Now()
	bs := encBatch
	for start := 0; start < len(order); start += bs {
		chunk := order[start:min(start+bs, len(order))]
		T := (len(ids[chunk[0]]) + 15) / 16 * 16
		toks, lens := make([]int32, bs*T), make([]int32, bs)
		for r := range lens {
			lens[r] = 1 // padding rows: one <pad> token, so that the mean is defined
		}
		for r, i := range chunk {
			copy(toks[r*T:], ids[i])
			lens[r] = int32(len(ids[i]))
		}
		out := exec.MustCall1(tensors.FromFlatDataAndDimensions(toks, bs, T), tensors.FromFlatDataAndDimensions(lens, bs))
		flat := tensors.MustCopyFlatData[float32](out)
		_ = out.FinalizeAll()
		for r, i := range chunk {
			rows[i] = append([]float32(nil), flat[r*f.H:(r+1)*f.H]...)
		}
	}
	fmt.Printf("  features %-9s %6d EmbeddingGemma 2 vectors in %v\n", name, len(texts), time.Since(t0).Round(time.Second))
	if path != "" {
		must(saveRows(path, rows))
	}
	f.rows = append(f.rows, rows...)
}

func saveRows(path string, rows [][]float32) error {
	fh, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fh.Close()
	if err := binary.Write(fh, binary.LittleEndian, []int64{int64(len(rows)), int64(len(rows[0]))}); err != nil {
		return err
	}
	for _, r := range rows {
		if err := binary.Write(fh, binary.LittleEndian, r); err != nil {
			return err
		}
	}
	return nil
}

func loadRows(path string, n, H int) ([][]float32, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	hdr := make([]int64, 2)
	if err := binary.Read(fh, binary.LittleEndian, hdr); err != nil {
		return nil, err
	}
	if int(hdr[0]) != n || int(hdr[1]) != H {
		return nil, fmt.Errorf("%s holds %dx%d, want %dx%d", path, hdr[0], hdr[1], n, H)
	}
	buf := make([]float32, n*H)
	if err := binary.Read(fh, binary.LittleEndian, buf); err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = buf[i*H : (i+1)*H]
	}
	return rows, nil
}
