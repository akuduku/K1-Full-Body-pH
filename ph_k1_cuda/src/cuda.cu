// CUDA launcher: one thread = one robot, the command path and all `nsub` plant steps inside one launch.
// The per-robot work buffers live in local memory (L1/L2-cached); the constants block is read through
// the read-only cache.
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "ph_k1/launch.hpp"

namespace ph {

template <class T>
__global__ void step_kernel(const ModelData<T>* __restrict__ md, BatchT<T> b) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= b.B) return;
    Work<T> w;
    run_robot(*md, b, r, w);
}

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ph_k1_native CUDA ") + what + ": " + cudaGetErrorString(e));
}

template <class T>
void run_cuda(const ModelData<T>* md_device, int /*ncontact*/, const BatchT<T>& b, void* stream) {
    if (b.B <= 0) return;
    // the work buffers are large (~24 kB float, ~48 kB double per thread): make sure the per-thread
    // stack/local-memory limit covers them
    static bool sized = false;
    if (!sized) {
        cudaFuncAttributes attr;
        check(cudaFuncGetAttributes(&attr, step_kernel<T>), "cudaFuncGetAttributes");
        size_t limit = 0;
        check(cudaDeviceGetLimit(&limit, cudaLimitStackSize), "cudaDeviceGetLimit");
        const size_t need = attr.localSizeBytes + 4096;
        if (limit < need) check(cudaDeviceSetLimit(cudaLimitStackSize, need), "cudaDeviceSetLimit");
        sized = true;
    }
    const int threads = 64;
    const int blocks = (b.B + threads - 1) / threads;
    step_kernel<T><<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(md_device, b);
    check(cudaGetLastError(), "launch");
}

template void run_cuda<float>(const ModelData<float>*, int, const BatchT<float>&, void*);
template void run_cuda<double>(const ModelData<double>*, int, const BatchT<double>&, void*);

}  // namespace ph
