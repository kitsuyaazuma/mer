// Similarity scores BFCL tools by the cosine between the request and each tool description, with the frozen Gemma 3 270M
// or EmbeddingGemma 2 and no training.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"

	"github.com/kitsuyaazuma/mer/internal/backbone"
	"github.com/kitsuyaazuma/mer/internal/embgemma"
	"github.com/kitsuyaazuma/mer/internal/tokspan"
	"github.com/kitsuyaazuma/mer/internal/toolbench"
)

var (
	flagEncoder  = flag.String("encoder", "gemma", "gemma (the frozen Gemma 3 270M, mean of the hidden states) or embgemma2 (EmbeddingGemma 2, retrieval prompts)")
	flagCacheDir = flag.String("cache_dir", "", "directory where the vectors are kept between runs (none if empty)")
	flagDumpCos  = flag.String("dump_cos", "", "CSV file for the cosine of every tool of the Index cases: id, tool, gold, cosine (none if empty)")
)

const usage = `Usage: go run ./cmd/similarity [flags]

Answers BFCL without training: a tool is called when the request is similar enough to its description (cosine).

Examples (the commands of the article):
  # the frozen Gemma 3 270M; the CSV keeps the cosine of every tool of the Index cases
  go run ./cmd/similarity -cache_dir ./cache -dump_cos results/similarity_gemma_cos.csv
  # EmbeddingGemma 2
  go run ./cmd/similarity -encoder embgemma2 -cache_dir ./cache

Flags:
`

const (
	bfclDir   = "data/bfcl"
	toolChars = 1500        // tool texts are cut to this many bytes
	qHead     = "STATE:\n"  // put before the request
	dHead     = "OPTION:\n" // put before each tool
)

func must(err error) {
	if err != nil {
		log.Fatalf("%+v", err)
	}
}
func must1[T any](v T, err error) T              { must(err); return v }
func must2[A, B any](a A, b B, err error) (A, B) { must(err); return a, b }

func main() {
	flag.Usage = func() {
		fmt.Fprint(flag.CommandLine.Output(), usage)
		flag.PrintDefaults()
	}
	flag.Parse()
	t0 := time.Now()
	backend := compute.MustNew()
	qPre, dPre := qHead, dHead
	if *flagEncoder == "embgemma2" {
		qPre, dPre = embgemma.PromptQuery, embgemma.PromptDocument
	}

	// Every category is read: the thresholds tried are quantiles of the cosines of all these cases.
	cats := append(append([]string{}, toolbench.CallCategories...), toolbench.IrrelevantCategories...)
	var cases []toolbench.Case
	for _, c := range cats {
		for _, x := range must1(toolbench.LoadBFCL(bfclDir, c)) {
			if len(x.Gold) > 0 && !x.GoldInTools() {
				fmt.Printf("skipped %s: gold %v is not among its tools\n", x.ID, x.Gold)
				continue
			}
			cases = append(cases, x)
		}
	}

	// Each distinct text is turned into a vector once: the request after qPre, each tool after dPre.
	idx := map[string]int{}
	var texts []string
	add := func(s string) int {
		if i, ok := idx[s]; ok {
			return i
		}
		idx[s] = len(texts)
		texts = append(texts, s)
		return idx[s]
	}
	qRow := make([]int, len(cases))
	tRows := make([][]int, len(cases))
	for i, c := range cases {
		qRow[i] = add(qPre + c.Query)
		for _, t := range c.Tools {
			tRows[i] = append(tRows[i], add(dPre+t.Text(toolChars)))
		}
	}
	var vecs [][]float32
	switch *flagEncoder {
	case "gemma":
		vecs = gemmaVectors(backend, must1(backbone.Load(backend, backbone.DefaultRepoID)), texts, "similarity-gemma")
	case "embgemma2":
		vecs = embgemmaVectors(backend, must1(embgemma.Load(backend, embgemma.Repo)), texts, "similarity")
	default:
		log.Fatalf("unknown -encoder %q", *flagEncoder)
	}

	// cos[i][j] is the cosine between the request of case i and its tool j.
	cos := make([][]float64, len(cases))
	for i := range cases {
		for _, r := range tRows[i] {
			cos[i] = append(cos[i], dot(vecs[qRow[i]], vecs[r]))
		}
	}

	byCat := map[string][]int{}
	for i, c := range cases {
		byCat[c.Category] = append(byCat[c.Category], i)
	}
	fmt.Printf("\nBFCL v3 (commit 916260d), encoder %s, prompt=%q / %q, tool text <= %d bytes\n", *flagEncoder, qPre, dPre, toolChars)
	// multiple picks the threshold; the other three are the Index cases that are scored.
	fmt.Printf("\n%-14s %6s %8s %8s\n", "category", "cases", "tools", "top-1")
	for _, c := range []string{"simple", "multiple", "live_simple", "live_multiple"} {
		nt, ok := 0, 0
		for _, i := range byCat[c] {
			nt += len(cases[i].Tools)
			if cases[i].Tools[argmax(cos[i])].Name == cases[i].Gold[0] {
				ok++
			}
		}
		n := len(byCat[c])
		fmt.Printf("%-14s %6d %8.1f %8.3f\n", c, n, float64(nt)/float64(n), float64(ok)/float64(n))
	}

	var index []int
	for _, c := range toolbench.IndexCategories {
		index = append(index, byCat[c]...)
	}
	// perTool says yes to the tools whose cosine is at least tau; a case is right when every tool is.
	perTool := func(i int, tau float64) bool {
		c := cases[i]
		for j, t := range c.Tools {
			if (cos[i][j] >= tau) != (t.Name == c.Gold[0]) {
				return false
			}
		}
		return true
	}
	bestTau, best := 0.0, -1.0
	for _, tau := range candidates(cos) {
		if a := accuracy(perTool, tau, byCat["multiple"]); a > best {
			bestTau, best = tau, a
		}
	}
	fmt.Printf("\nper-tool yes/no, tau chosen on BFCL multiple (%d cases, outside the Index): tau=%.4f  multiple %.3f  Index (all %d) %.4f\n",
		len(byCat["multiple"]), bestTau, best, len(index), accuracy(perTool, bestTau, index))

	allYes := 0
	for _, i := range index {
		if len(cases[i].Tools) == 1 {
			allYes++
		}
	}
	fmt.Printf("reference: 'yes to every tool' on the Index cases = %.4f (%d / %d)\n", float64(allYes)/float64(len(index)), allYes, len(index))

	top1, multi := 0, 0
	chance := 0.0 // the share a random pick of one tool gets right
	var dumpRows []string
	for _, i := range index {
		for j, t := range cases[i].Tools {
			gold := 0
			if t.Name == cases[i].Gold[0] {
				gold = 1
			}
			dumpRows = append(dumpRows, fmt.Sprintf("%s,%s,%d,%.6f", cases[i].ID, t.Name, gold, cos[i][j]))
		}
		if len(cases[i].Tools) > 1 {
			multi++
			chance += 1 / float64(len(cases[i].Tools))
			if cases[i].Tools[argmax(cos[i])].Name == cases[i].Gold[0] {
				top1++
			}
		}
	}
	fmt.Printf("multi-tool cases where the gold tool is the most similar of its request: %d / %d (%.3f, chance %.3f)\n",
		top1, multi, float64(top1)/float64(multi), chance/float64(multi))
	if *flagDumpCos != "" {
		must(os.WriteFile(*flagDumpCos, []byte("id,tool,gold,cosine\n"+strings.Join(dumpRows, "\n")+"\n"), 0o644))
	}
	fmt.Printf("total %v\n", time.Since(t0).Round(time.Second))
}

func argmax(xs []float64) int {
	j := 0
	for i, x := range xs {
		if x > xs[j] {
			j = i
		}
	}
	return j
}

func dot(a, b []float32) float64 {
	s := 0.0
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// accuracy is the share of the cases ids that rule gets right with the threshold tau.
func accuracy(rule func(int, float64) bool, tau float64, ids []int) float64 {
	ok := 0
	for _, i := range ids {
		if rule(i, tau) {
			ok++
		}
	}
	return float64(ok) / float64(max(1, len(ids)))
}

// candidates are the thresholds tried: 400 quantiles of all the cosines, and one below and one above them all.
func candidates(cos [][]float64) []float64 {
	var all []float64
	for _, r := range cos {
		all = append(all, r...)
	}
	sort.Float64s(all)
	out := []float64{all[0] - 1e-6, all[len(all)-1] + 1e-6}
	for k := 1; k < 400; k++ {
		out = append(out, all[k*len(all)/400])
	}
	return out
}

// embgemmaVectors returns the EmbeddingGemma 2 vectors of texts, cached under name.
func embgemmaVectors(backend compute.Backend, em *embgemma.Model, texts []string, name string) [][]float32 {
	h := fnv.New64a()
	for _, t := range texts {
		fmt.Fprintf(h, "%s\x00", t)
	}
	path := ""
	if *flagCacheDir != "" {
		must(os.MkdirAll(*flagCacheDir, 0o755))
		path = fmt.Sprintf("%s/feat-%s-%x.f32", *flagCacheDir, name, h.Sum64())
		if rows := loadRows(path, len(texts), 768); rows != nil {
			fmt.Printf("%d vectors loaded from cache\n", len(rows))
			return rows
		}
	}
	exec := model.MustNewExec(backend, em.Store, func(scope *model.Scope, tokens, lengths *Node) *Node { return em.Embed(scope, tokens, lengths) })
	ids := make([][]int32, len(texts))
	order := make([]int, len(texts))
	for i, t := range texts {
		ids[i], order[i] = em.Encode(t), i
		if len(ids[i]) > 1024 {
			ids[i] = ids[i][:1024]
		}
	}
	sort.SliceStable(order, func(a, b int) bool { return len(ids[order[a]]) > len(ids[order[b]]) })
	rows := make([][]float32, len(texts))
	t0 := time.Now()
	for lo := 0; lo < len(order); {
		T := bucket(len(ids[order[lo]]))
		bs := max(1, min(64, 32768/T)) // about the same number of tokens per batch; one graph per bucket
		chunk := order[lo:min(lo+bs, len(order))]
		toks, lens := make([]int32, bs*T), make([]int32, bs)
		for r := range lens {
			lens[r] = 1
		}
		for r, i := range chunk {
			copy(toks[r*T:], ids[i])
			lens[r] = int32(len(ids[i]))
		}
		out := exec.MustCall1(tensors.FromFlatDataAndDimensions(toks, bs, T), tensors.FromFlatDataAndDimensions(lens, bs))
		flat := tensors.MustCopyFlatData[float32](out)
		_ = out.FinalizeAll()
		for r, i := range chunk {
			rows[i] = append([]float32(nil), flat[r*768:(r+1)*768]...)
		}
		lo += len(chunk)
	}
	fmt.Printf("%d EmbeddingGemma 2 vectors in %v\n", len(texts), time.Since(t0).Round(time.Second))
	if path != "" {
		f := must1(os.Create(path))
		must(binary.Write(f, binary.LittleEndian, []int64{int64(len(rows)), 768}))
		for _, r := range rows {
			must(binary.Write(f, binary.LittleEndian, r))
		}
		must(f.Close())
	}
	return rows
}

// gemmaVectors returns the L2-normalized mean of the hidden states of each text, over the tokens after its head.
func gemmaVectors(backend compute.Backend, bb *backbone.Backbone, texts []string, name string) [][]float32 {
	H := bb.HiddenDim
	h := fnv.New64a()
	for _, t := range texts {
		fmt.Fprintf(h, "%s\x00", t)
	}
	path := ""
	if *flagCacheDir != "" {
		must(os.MkdirAll(*flagCacheDir, 0o755))
		path = fmt.Sprintf("%s/feat-%s-%x.f32", *flagCacheDir, name, h.Sum64())
		if rows := loadRows(path, len(texts), H); rows != nil {
			fmt.Printf("%d vectors loaded from cache\n", len(rows))
			return rows
		}
	}
	exec := model.MustNewExec(backend, bb.Store, func(scope *model.Scope, tokens *Node) *Node { return bb.HiddenStates(scope, tokens) })
	exec.SetMaxCache(128)
	ids := make([][]int32, len(texts))
	spans := make([][2]int32, len(texts))
	order := make([]int, len(texts))
	for i, t := range texts {
		head := qHead
		if strings.HasPrefix(t, dHead) {
			head = dHead
		}
		var sp [][2]int32
		ids[i], sp = must2(tokspan.Encode(bb.Tok, t, [][2]int{{len(head), len(t)}}))
		if len(ids[i]) > 1024 {
			ids[i] = ids[i][:1024]
		}
		spans[i] = [2]int32{sp[0][0], min(sp[0][1], int32(len(ids[i])))}
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(ids[order[a]]) > len(ids[order[b]]) })
	rows := make([][]float32, len(texts))
	t0 := time.Now()
	for start := 0; start < len(order); {
		T := bucket(len(ids[order[start]]))
		bs := max(1, min(32, 16384/T)) // about the same number of tokens per batch
		chunk := order[start:min(start+bs, len(order))]
		toks := make([]int32, bs*T)
		for r, i := range chunk {
			copy(toks[r*T:], ids[i])
		}
		out := exec.MustCall1(tensors.FromFlatDataAndDimensions(toks, bs, T))
		flat := tensors.MustCopyFlatData[float32](out)
		_ = out.FinalizeAll()
		for r, i := range chunk {
			v := make([]float32, H)
			for t := spans[i][0]; t < spans[i][1]; t++ {
				for j := range v {
					v[j] += flat[(r*T+int(t))*H+j]
				}
			}
			n := 0.0
			for _, x := range v {
				n += float64(x) * float64(x)
			}
			for j := range v {
				v[j] /= float32(math.Sqrt(n) + 1e-12)
			}
			rows[i] = v
		}
		start += len(chunk)
	}
	fmt.Printf("%d Gemma vectors in %v\n", len(texts), time.Since(t0).Round(time.Second))
	if path != "" {
		f := must1(os.Create(path))
		must(binary.Write(f, binary.LittleEndian, []int64{int64(len(rows)), int64(H)}))
		for _, r := range rows {
			must(binary.Write(f, binary.LittleEndian, r))
		}
		must(f.Close())
	}
	return rows
}

// bucket keeps the number of compiled graphs small: GoMLX keeps at most 32 per Exec.
func bucket(n int) int {
	for _, b := range []int{32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024} {
		if n <= b {
			return b
		}
	}
	return (n + 255) / 256 * 256
}

func loadRows(path string, n, H int) [][]float32 {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	hdr := make([]int64, 2)
	if binary.Read(f, binary.LittleEndian, hdr) != nil || int(hdr[0]) != n || int(hdr[1]) != H {
		return nil
	}
	buf := make([]float32, n*H)
	if binary.Read(f, binary.LittleEndian, buf) != nil {
		return nil
	}
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = buf[i*H : (i+1)*H]
	}
	return rows
}
