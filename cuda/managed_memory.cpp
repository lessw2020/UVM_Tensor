
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>
#include <cassert>
#include <iostream>

at::Tensor getManagedTensor(size_t nb_bytes, c10::IntArrayRef sizes) {

    float* cuda_ptr;
    cudaMallocManaged(&cuda_ptr, nb_bytes);

    at::Tensor tensor = at::for_blob((void*)cuda_ptr, sizes)
             .deleter([](void *ptr) {
               cudaFree(ptr);
             })
             .options(at::device(at::kCUDA).dtype(at::kFloat))
             .make_tensor();

    return tensor;
}
