package gocublas

import (
	"testing"

	"github.com/gonum/blas/native"
	"github.com/gonum/blas/testblas"
)

func TestDgemm(t *testing.T) {
	impl, err := NewFloat64(native.Implementation{})
	if err != nil {
		t.Fatal(err)
	}
	defer impl.Destroy()
	testblas.TestDgemm(t, impl)
}
