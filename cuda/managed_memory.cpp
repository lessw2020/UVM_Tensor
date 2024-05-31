
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>
#include <cassert>
#include <iostream>

#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}
//Create a Unified Memory Tensor with built in delete function
//Original code from AlbanD and Jane
at::Tensor getManagedTensor(size_t nb_bytes, c10::IntArrayRef sizes) {

    float* cuda_ptr;

    CUDA_CHECK_RETURN(cudaMallocManaged(&cuda_ptr, nb_bytes));


    at::Tensor tensor = at::for_blob((void*)cuda_ptr, sizes)
             .deleter([](void *ptr) {
               cudaFree(ptr);
             })
             .options(at::device(at::kCUDA).dtype(at::kFloat)) //.memory_format(at::MemoryFormat::Contiguous))
             .make_tensor();

    return tensor; //, cuda_ptr;
}

void cuda_prefetch(at::Tensor in_tensor) //, size_t bytes, int device)
{
    void* ptr = in_tensor.data_ptr();
    size_t bytes = in_tensor.numel() * in_tensor.element_size();
    int device = in_tensor.get_device();

    CUDA_CHECK_RETURN(cudaMemPrefetchAsync(ptr, bytes, device, 0));
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("getManagedTensor", &getManagedTensor, "get um managed tensor (CUDA)");
  m.def("cuda_prefetch", &cuda_prefetch, "prefetch um managed tensor to given device");
}
