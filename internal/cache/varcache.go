// Package cache stores the frozen backbone's hidden states so that they are computed only once.
package cache

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/edsrzf/mmap-go"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
)

// VarCache keeps the hidden states of only the real tokens of each example and zero-pads batches on Gather. The
// values are in memory, or in a file mapped into memory, so that the operating system reads them from the disk as
// batches need them and can drop them again when memory runs short.
type VarCache struct {
	H    int
	Len  []int // tokens of each example
	off  []int
	data []byte // bfloat16 values, token after token

	path string    // the file of a mapped cache
	f    *os.File  // nil for a cache in memory
	mm   mmap.MMap // the whole file: the header, then data
}

func layout(lens []int) (off []int, tokens int) {
	off = make([]int, len(lens))
	for i, l := range lens {
		off[i] = tokens
		tokens += l
	}
	return off, tokens
}

// headerSize is the bytes before the data: n and H (int64), then the n lengths (int32).
func headerSize(n int) int { return 16 + 4*n }

// NewVarCache allocates a cache in memory for examples of the given lengths.
func NewVarCache(lens []int, H int) *VarCache {
	off, tokens := layout(lens)
	return &VarCache{H: H, Len: lens, off: off, data: make([]byte, tokens*H*2)}
}

// CreateVarCache makes a cache in a new file next to path, to be filled with Put; Commit moves it to path.
func CreateVarCache(path string, lens []int, H int) (*VarCache, error) {
	off, tokens := layout(lens)
	c := &VarCache{H: H, Len: lens, off: off, path: path}
	f, err := os.Create(path + ".tmp")
	if err != nil {
		return nil, err
	}
	hdr := make([]byte, headerSize(len(lens)))
	binary.LittleEndian.PutUint64(hdr, uint64(len(lens)))
	binary.LittleEndian.PutUint64(hdr[8:], uint64(H))
	for i, l := range lens {
		binary.LittleEndian.PutUint32(hdr[16+4*i:], uint32(l))
	}
	if _, err = f.Write(hdr); err == nil {
		err = f.Truncate(int64(len(hdr) + tokens*H*2))
	}
	if err != nil {
		f.Close()
		return nil, err
	}
	if err := c.mapFile(f, mmap.RDWR, len(hdr)); err != nil {
		return nil, err
	}
	return c, nil
}

// Commit writes the cache made by CreateVarCache to the disk under its path and maps it again read only.
func (c *VarCache) Commit() error {
	if err := c.mm.Flush(); err != nil {
		return err
	}
	if err := c.Close(); err != nil {
		return err
	}
	if err := os.Rename(c.path+".tmp", c.path); err != nil {
		return err
	}
	opened, err := OpenVarCache(c.path, c.Len, c.H)
	if err != nil {
		return err
	}
	*c = *opened
	return nil
}

// OpenVarCache maps a cache file and checks it holds examples of exactly these lengths.
func OpenVarCache(path string, lens []int, H int) (*VarCache, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	hdr := make([]byte, headerSize(len(lens)))
	if _, err := io.ReadFull(f, hdr[:16]); err != nil {
		f.Close()
		return nil, err
	}
	n, h := int(binary.LittleEndian.Uint64(hdr)), int(binary.LittleEndian.Uint64(hdr[8:]))
	if n != len(lens) || h != H {
		f.Close()
		return nil, fmt.Errorf("%s holds (n=%d, H=%d), want (%d, %d)", path, n, h, len(lens), H)
	}
	if _, err := io.ReadFull(f, hdr[16:]); err != nil {
		f.Close()
		return nil, err
	}
	for i, l := range lens {
		if got := int(binary.LittleEndian.Uint32(hdr[16+4*i:])); got != l {
			f.Close()
			return nil, fmt.Errorf("%s: example %d has %d tokens, want %d", path, i, got, l)
		}
	}
	off, tokens := layout(lens)
	c := &VarCache{H: H, Len: lens, off: off, path: path}
	if st, err := f.Stat(); err != nil || st.Size() != int64(len(hdr)+tokens*H*2) {
		f.Close()
		return nil, fmt.Errorf("%s is not complete", path)
	}
	if err := c.mapFile(f, mmap.RDONLY, len(hdr)); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *VarCache) mapFile(f *os.File, prot int, hdr int) error {
	mm, err := mmap.Map(f, prot, 0)
	if err != nil {
		f.Close()
		return err
	}
	c.f, c.mm, c.data = f, mm, mm[hdr:]
	return nil
}

// Close unmaps a cache in a file; a cache in memory needs no Close.
func (c *VarCache) Close() error {
	if c.f == nil {
		return nil
	}
	err := c.mm.Unmap()
	if cerr := c.f.Close(); err == nil {
		err = cerr
	}
	c.f, c.mm, c.data = nil, nil, nil
	return err
}

// Put stores the first Len tokens of each row of hidden [batch, T, H] (bfloat16) for the examples idx.
func (c *VarCache) Put(idx []int, hidden *tensors.Tensor) error {
	T := hidden.Shape().Dimensions[1]
	row := c.H * 2
	return hidden.ConstBytes(func(b []byte) {
		for r, e := range idx {
			copy(c.data[c.off[e]*row:(c.off[e]+c.Len[e])*row], b[r*T*row:(r*T+c.Len[e])*row])
		}
	})
}

// Gather builds a [len(idx), T, H] bfloat16 batch, every example padded with zeros to T tokens.
func (c *VarCache) Gather(idx []int, T int) (*tensors.Tensor, error) {
	t := tensors.FromShape(shapes.Make(dtypes.BFloat16, len(idx), T, c.H))
	row := c.H * 2
	err := t.MutableBytes(func(dst []byte) {
		clear(dst)
		for r, e := range idx {
			copy(dst[r*T*row:], c.data[c.off[e]*row:(c.off[e]+c.Len[e])*row])
		}
	})
	return t, err
}
