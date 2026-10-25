// Package massive downloads Amazon's MASSIVE dataset (ja-JP) and turns its utterances into decision prompts.
package massive

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// URL of MASSIVE 1.1 (CC BY 4.0).
const URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"

// Utterance is one line of ja-JP.jsonl (only the fields we use).
type Utterance struct {
	ID        string `json:"id"`
	Partition string `json:"partition"` // "train", "dev" or "test"
	Scenario  string `json:"scenario"`
	Intent    string `json:"intent"`
	Text      string `json:"utt"`
	Annot     string `json:"annot_utt"` // utt with each slot as [label : entity], spaces added around slots: "[date : 金曜日] の [time : 午前九時] に起こしてください"
}

// EnsureJA returns the path of dir/ja-JP.jsonl, downloading and extracting it from the MASSIVE tarball if needed.
func EnsureJA(dir string) (string, error) {
	path := filepath.Join(dir, "ja-JP.jsonl")
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	fmt.Printf("downloading %s ...\n", URL)
	resp, err := http.Get(URL)
	if err != nil {
		return "", fmt.Errorf("downloading MASSIVE: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("downloading MASSIVE: %s", resp.Status)
	}
	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return "", err
	}
	tr := tar.NewReader(gz)
	found := false
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return "", err
		}
		var out string
		switch filepath.Base(hdr.Name) {
		case "ja-JP.jsonl":
			out, found = path, true
		case "LICENSE":
			out = filepath.Join(dir, "LICENSE")
		default:
			continue
		}
		f, err := os.Create(out)
		if err != nil {
			return "", err
		}
		if _, err := io.Copy(f, tr); err != nil {
			f.Close()
			return "", err
		}
		if err := f.Close(); err != nil {
			return "", err
		}
	}
	if !found {
		return "", errors.New("ja-JP.jsonl not found in the MASSIVE tarball")
	}
	return path, nil
}

// Load reads all utterances of a MASSIVE jsonl file.
func Load(path string) ([]Utterance, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var utts []Utterance
	dec := json.NewDecoder(f)
	for {
		var u Utterance
		err := dec.Decode(&u)
		if errors.Is(err, io.EOF) {
			return utts, nil
		}
		if err != nil {
			return nil, fmt.Errorf("parsing %s: %w", path, err)
		}
		utts = append(utts, u)
	}
}
