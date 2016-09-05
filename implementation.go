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

// Dgemm uses cuBLAS to multiply two matrices.
func (f *float64Impl) Dgemm(tA blas.Transpose, tB blas.Transpose, m int, n int, k int,
	alpha float64, a []float64, lda int, b []float64, ldb int, beta float64,
	c []float64, ldc int) {
	validateGemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	err := f.blasHandle.doubleMatMult(tA != blas.NoTrans, tB != blas.NoTrans, m, n, k,
		alpha, a, lda, b, ldb, beta, c, ldc)
	if err != nil {
		panic(err)
	}
}

func validateGemm(tA blas.Transpose, tB blas.Transpose, m int, n int, k int,
	alpha float64, a []float64, lda int, b []float64, ldb int, beta float64,
	c []float64, ldc int) {
	// Checking code taken from https://github.com/gonum/blas/blob/447542bc23b84f8daa64470951a57bb6713d15a1/cgo/blas.go#L2848-L2876
	// Said code is under the following license:
	//
	// Copyright ©2013 The gonum Authors. All rights reserved.
	//
	// Redistribution and use in source and binary forms, with or without
	// modification, are permitted provided that the following conditions are met:
	//     * Redistributions of source code must retain the above copyright
	//       notice, this list of conditions and the following disclaimer.
	//     * Redistributions in binary form must reproduce the above copyright
	//       notice, this list of conditions and the following disclaimer in the
	//       documentation and/or other materials provided with the distribution.
	//     * Neither the name of the gonum project nor the names of its authors and
	//       contributors may be used to endorse or promote products derived from this
	//       software without specific prior written permission.
	//
	// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
	// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
	// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
	// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var rowA, colA, rowB, colB int
	if tA == blas.NoTrans {
		rowA, colA = m, k
	} else {
		rowA, colA = k, m
	}
	if tB == blas.NoTrans {
		rowB, colB = k, n
	} else {
		rowB, colB = n, k
	}
	if lda*(rowA-1)+colA > len(a) || lda < max(1, colA) {
		panic("blas: index of a out of range")
	}
	if ldb*(rowB-1)+colB > len(b) || ldb < max(1, colB) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
}
