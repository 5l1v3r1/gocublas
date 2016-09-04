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
*/
import "C"
import (
	"errors"
	"unsafe"
)

var ErrHandleCreation = errors.New("failed to create cuBLAS handle")

type blasHandle struct {
	handlePtr unsafe.Pointer
	destroyed bool
}

func newBLASHandle() (*blasHandle, error) {
	handle := C.gocublas_new_handle()
	if C.gocublas_is_null(handle) == C.int(0) {
		return nil, ErrHandleCreation
	}
	return &blasHandle{handlePtr: handle}, nil
}

func (h *blasHandle) Destroy() {
	if !h.destroyed {
		h.destroyed = true
		C.gocublas_destroy_handle(h.handlePtr)
	}
}
