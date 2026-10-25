// Serve answers System One requests over HTTP, POST /v1/systemone as Ollama's System One API, with a Mer saved by
// cmd/mer -save.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"

	"github.com/kitsuyaazuma/mer/internal/mer"
	"github.com/kitsuyaazuma/mer/internal/systemone"
)

var (
	flagModel = flag.String("model", "", "directory of the Mer saved by cmd/mer -save")
	flagAddr  = flag.String("addr", "localhost:11434", "address to listen on")
)

const usage = `Usage: go run ./cmd/serve -model DIR [flags]

Answers POST /v1/systemone (https://docs.ollama.com/api/systemone) with the Mer saved in DIR, whatever the model
named in the request. Example:

  curl http://localhost:11434/v1/systemone -d '{"model": "mer", "state": "Our checkout has returned 500 errors since 9am.",
    "questions": {"label": {"type": "choice", "instructions": "Which label fits this ticket?",
      "criteria": {"billing": "Payments and refunds", "bug": "Software errors", "account": "Login and account access"}}}}'

Flags:
`

// maxBody is the largest request read, as a local Ollama server reads.
const maxBody = 32 << 20

func main() {
	flag.Usage = func() {
		fmt.Fprint(flag.CommandLine.Output(), usage)
		flag.PrintDefaults()
	}
	flag.Parse()
	if *flagModel == "" {
		flag.Usage()
		log.Fatal("-model is needed")
	}
	m, err := mer.Load(compute.MustNew(), *flagModel)
	if err != nil {
		log.Fatalf("loading %s: %+v", *flagModel, err)
	}
	s := &server{m: m}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/systemone", s.systemOne)
	log.Printf("serving %s (%s on %s) at http://%s/v1/systemone", *flagModel, m.Meta.Design, m.Meta.Backbone, *flagAddr)
	log.Fatal(http.ListenAndServe(*flagAddr, mux))
}

type server struct {
	mu sync.Mutex // one request at a time on the backend
	m  *mer.Model
}

func (s *server) systemOne(w http.ResponseWriter, req *http.Request) {
	var wr systemone.WireRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, maxBody)).Decode(&wr); err != nil {
		status := http.StatusBadRequest
		if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
			status = http.StatusRequestEntityTooLarge
		}
		writeError(w, status, err)
		return
	}
	if strings.TrimSpace(wr.Model) == "" {
		writeError(w, http.StatusBadRequest, errors.New("model is required"))
		return
	}
	r, err := systemone.Parse(wr)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	s.mu.Lock()
	logits, tokens, err := s.m.Answer(r)
	s.mu.Unlock()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, systemone.Response{Model: wr.Model, Answers: systemone.Answers(r, logits), Usage: systemone.Usage{InputTokens: tokens}})
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("writing the response: %v", err)
	}
}
