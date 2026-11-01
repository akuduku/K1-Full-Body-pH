// A batch of robots as raw row-major pointers (B rows each), and the two launchers that run
// `nsub` plant steps for every robot: the CPU loop (src/cpu.cpp) and the CUDA kernel (src/cuda.cu).
#pragma once

#include "step.hpp"

namespace ph {

template <class T>
struct BatchT {
    int B = 0;
    int nsub = 1;
    T* q = nullptr;
    T* p = nullptr;
    T* xi = nullptr;
    T* lam = nullptr;
    T* ledger = nullptr;
    const T* volts = nullptr;
    const T* jtau = nullptr;
    T* q_star = nullptr;
    const T* target = nullptr;
    const T* kp = nullptr;
    const T* kd = nullptr;
    T* out_nu = nullptr;
    T* out_i = nullptr;
    T* out_tau = nullptr;
    T* out_v = nullptr;
    T* out_fc = nullptr;
    uint8_t* out_active = nullptr;
    uint8_t* out_slip = nullptr;
};

template <class T>
PH_HD StepIO<T> make_io(const BatchT<T>& b, int r, int nc) {
    StepIO<T> io;
    io.q = b.q + r * NQ;
    io.p = b.p + r * NV;
    io.xi = b.xi + r * nc * 2;
    io.lam = b.lam + r * NJ;
    io.ledger = b.ledger + r * LEDGER;
    io.volts = b.volts ? b.volts + r * NJ : nullptr;
    io.jtau = b.jtau ? b.jtau + r * NJ : nullptr;
    io.q_star = b.q_star ? b.q_star + r * NJ : nullptr;
    io.target = b.target ? b.target + r * NJ : nullptr;
    io.kp = b.kp ? b.kp + r * NJ : nullptr;
    io.kd = b.kd ? b.kd + r * NJ : nullptr;
    io.out_nu = b.out_nu ? b.out_nu + r * NV : nullptr;
    io.out_i = b.out_i ? b.out_i + r * NJ : nullptr;
    io.out_tau = b.out_tau ? b.out_tau + r * NJ : nullptr;
    io.out_v = b.out_v ? b.out_v + r * NJ : nullptr;
    io.out_fc = b.out_fc ? b.out_fc + r * nc * 3 : nullptr;
    io.out_active = b.out_active ? b.out_active + r * nc : nullptr;
    io.out_slip = b.out_slip ? b.out_slip + r * nc : nullptr;
    return io;
}

template <class T> void run_cpu(const ModelData<T>& m, const BatchT<T>& b);
#ifdef PH_K1_WITH_CUDA
// md_device: the ModelData block in GPU memory; all batch pointers are GPU pointers; `stream` is a cudaStream_t
template <class T> void run_cuda(const ModelData<T>* md_device, int ncontact, const BatchT<T>& b, void* stream);
#endif

}  // namespace ph
