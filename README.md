# gocublas

This package makes it possible to use NVIDIA's [cuBLAS](https://developer.nvidia.com/cublas) library with the [gonum/blas](https://github.com/gonum/blas) package.

# Usage

In order for this to work, your compiler and the dynamic linker need to know where to find CUDA. In my case, CUDA is installed in `/Developer/NVIDIA/CUDA-7.5`. You can change this to be the path of your own installation. Simply set the following environment variables, and you should be able to use this package (while those variables are set):

```
$ export CUDA_PATH="/Developer/NVIDIA/CUDA-7.5"
$ export DYLD_LIBRARY_PATH="$CUDA_PATH/lib":$DYLD_LIBRARY_PATH
$ export CPATH="$CUDA_PATH/include/"
$ export CGO_LDFLAGS="$CUDA_PATH/lib/libcublas.dylib $CUDA_PATH/lib/libcudart.dylib"
```