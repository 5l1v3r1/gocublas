package gocublas

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Cocoa

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"

void * gocublas_new_handle() {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUDA_SUCCESS) {
        return NULL;
    }
    return handle;
}

void gocublas_destroy_handle(void * handle) {
    cublasDestroy((cublasHandle_t)handle);
}

int gocublas_is_null(void * ptr) {
    return ptr == NULL;
}

int gocublas_dgemm(void * handle, int transA, int transB, int m, int n, int k,
    double alpha, double * matA, int lda, double * matB, int ldb, double beta,
    double * matC, int ldc) {
    void * mat1;
    void * mat2;
    void * mat3;
    if (cudaMalloc(&mat1, m*k*sizeof(double)) != cudaSuccess) {
        return 1;
    }
    if (cudaMalloc(&mat2, n*k*sizeof(double)) != cudaSuccess) {
        cudaFree(mat1);
        return 1;
    }
    if (cudaMalloc(&mat3, m*n*sizeof(double)) != cudaSuccess) {
        cudaFree(mat1);
        cudaFree(mat2);
        return 1;
    }
    if (cudaMemcpy(mat1, matA, m*k*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(mat3);
        return 2;
    }
    if (cudaMemcpy(mat2, matB, n*k*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(mat3);
        return 2;
    }
    if (cudaMemcpy(mat3, matC, m*n*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(mat3);
        return 2;
    }
    cublasOperation_t trans1, trans2;
    int ld1, ld2;
    if (transA) {
        ld1 = m;
        trans1 = CUBLAS_OP_T;
    } else {
        ld1 = k;
        trans1 = CUBLAS_OP_N;
    }
    if (transB) {
        ld2 = k;
        trans2 = CUBLAS_OP_T;
    } else {
        ld2 = n;
        trans2 = CUBLAS_OP_N;
    }
    cublasStatus_t s = cublasDgemm((cublasHandle_t)handle, trans2, trans1, n, m, k,
        &alpha, mat2, ld2, mat1, ld1, &beta, mat3, n);
    if (s != CUBLAS_STATUS_SUCCESS) {
        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(mat3);
        return 3;
    }
    cudaError_t e = cudaMemcpy(matC, mat3, m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(mat1);
    cudaFree(mat2);
    cudaFree(mat3);
    if (e != cudaSuccess) {
        return 2;
    }
    return 0;
}

int gocublas_sgemm(void * handle, int transA, int transB, int m, int n, int k,
    float alpha, float * matA, int lda, float * matB, int ldb, float beta,
    float * matC, int ldc) {
    void * mat1;
    void * mat2;
    void * mat3;
    if (cudaMalloc(&mat1, m*k*sizeof(float)) != cudaSuccess) {
        return 1;
    }
    if (cudaMalloc(&mat2, n*k*sizeof(float)) != cudaSuccess) {
        cudaFree(mat1);
        return 1;
    }
    if (cudaMalloc(&mat3, m*n*sizeof(float)) != cudaSuccess) {
        cudaFree(mat1);
        cudaFree(mat2);
        return 1;
    }
    if (cudaMemcpy(mat1, matA, m*k*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(mat3);
        return 2;
    }
    if (cudaMemcpy(mat2, matB, n*k*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(mat3);
        return 2;
    }
    if (cudaMemcpy(mat3, matC, m*n*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(mat3);
        return 2;
    }
    cublasOperation_t trans1, trans2;
    int ld1, ld2;
    if (transA) {
        ld1 = m;
        trans1 = CUBLAS_OP_T;
    } else {
        ld1 = k;
        trans1 = CUBLAS_OP_N;
    }
    if (transB) {
        ld2 = k;
        trans2 = CUBLAS_OP_T;
    } else {
        ld2 = n;
        trans2 = CUBLAS_OP_N;
    }
    cublasStatus_t s = cublasSgemm((cublasHandle_t)handle, trans2, trans1, n, m, k,
        &alpha, mat2, ld2, mat1, ld1, &beta, mat3, n);
    if (s != CUBLAS_STATUS_SUCCESS) {
        cudaFree(mat1);
        cudaFree(mat2);
        cudaFree(mat3);
        return 3;
    }
    cudaError_t e = cudaMemcpy(matC, mat3, m*n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(mat1);
    cudaFree(mat2);
    cudaFree(mat3);
    if (e != cudaSuccess) {
        return 2;
    }
    return 0;
}
*/
import "C"
import (
	"errors"
	"unsafe"
)

var (
	ErrHandleCreation = errors.New("failed to create cuBLAS handle")
	ErrMemoryAlloc    = errors.New("failed to allocate CUDA memory")
	ErrMemoryCopy     = errors.New("failed to transfer CUDA memory")
	ErrMatrixMultiply = errors.New("matrix multiplication failed")
)

type blasHandle struct {
	handlePtr unsafe.Pointer
	destroyed bool
}

func newBLASHandle() (*blasHandle, error) {
	handle := C.gocublas_new_handle()
	if C.gocublas_is_null(handle) != C.int(0) {
		return nil, ErrHandleCreation
	}
	return &blasHandle{handlePtr: handle}, nil
}

func (h *blasHandle) doubleMatMult(transA, transB bool, m, n, k int, alpha float64, matA []float64,
	lda int, matB []float64, ldb int, beta float64, matC []float64, ldc int) error {
	if h.destroyed {
		panic("cannot use a destroyed handle")
	}

	var transAInt, transBInt C.int
	if transA {
		transAInt = 1
	}
	if transB {
		transBInt = 1
	}

	res := C.gocublas_dgemm(h.handlePtr, transAInt, transBInt, C.int(m), C.int(n), C.int(k),
		C.double(alpha), (*C.double)(&matA[0]), C.int(lda), (*C.double)(&matB[0]),
		C.int(ldb), C.double(beta), (*C.double)(&matC[0]), C.int(ldc))
	switch res {
	case 0:
		return nil
	case 1:
		return ErrMemoryAlloc
	case 2:
		return ErrMemoryCopy
	case 3:
		return ErrMatrixMultiply
	}
	panic("unexpected return value from C API")
}

func (h *blasHandle) singleMatMult(transA, transB bool, m, n, k int, alpha float32, matA []float32,
	lda int, matB []float32, ldb int, beta float32, matC []float32, ldc int) error {
	if h.destroyed {
		panic("cannot use a destroyed handle")
	}

	var transAInt, transBInt C.int
	if transA {
		transAInt = 1
	}
	if transB {
		transBInt = 1
	}

	res := C.gocublas_sgemm(h.handlePtr, transAInt, transBInt, C.int(m), C.int(n), C.int(k),
		C.float(alpha), (*C.float)(&matA[0]), C.int(lda), (*C.float)(&matB[0]),
		C.int(ldb), C.float(beta), (*C.float)(&matC[0]), C.int(ldc))
	switch res {
	case 0:
		return nil
	case 1:
		return ErrMemoryAlloc
	case 2:
		return ErrMemoryCopy
	case 3:
		return ErrMatrixMultiply
	}
	panic("unexpected return value from C API")
}

func (h *blasHandle) Destroy() {
	if !h.destroyed {
		h.destroyed = true
		C.gocublas_destroy_handle(h.handlePtr)
	}
}

func max(i1, i2 int) int {
	if i1 > i2 {
		return i1
	} else {
		return i2
	}
}
