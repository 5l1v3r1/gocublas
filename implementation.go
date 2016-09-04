package gocublas

import "github.com/gonum/blas"

// A Destroyer is anything with some allocated context
// whose context should be disposed before program
// termination.
type Destroyer interface {
	Destroy()
}

type Implementation32 interface {
	Destroyer
	blas.Float32
}

type Implementation64 interface {
	Destroyer
	blas.Float64
}

// NewFloat32 wraps a BLAS implementation and overrides
// some of its methods to use cuBLAS.
// If this succeeds, you must Dispose() the result once
// you are done with it.
func NewFloat32(imp blas.Float32) (Implementation32, error) {
	handle, err := newBLASHandle()
	if err != nil {
		return nil, err
	}
	return &float32Impl{
		Float32:    imp,
		blasHandle: handle,
	}, nil
}

// NewFloat64 wraps a BLAS implementation and overrides
// some of its methods to use cuBLAS.
// If this succeeds, you must Dispose() the result once
// you are done with it.
func NewFloat64(imp blas.Float64) (Implementation64, error) {
	handle, err := newBLASHandle()
	if err != nil {
		return nil, err
	}
	return &float64Impl{
		Float64:    imp,
		blasHandle: handle,
	}, nil
}

type float32Impl struct {
	blas.Float32
	*blasHandle
}

type float64Impl struct {
	blas.Float64
	*blasHandle
}
