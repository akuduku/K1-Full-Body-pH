// A batch of robots as raw row-major pointers (B rows each), the per-robot driver loop shared by the CPU and
// the GPU, and the two launchers (src/cpu.cpp: OpenMP over robots; src/cuda.cu: one thread per robot).
//
// Two modes:
//   volts         nsub plant steps with the same volts (or the joint_torque test port): the plant alone
//   policy step   the robot's command path (ph_k1/firmware.py): the policy target goes through the firmware
//                 filter once, then for each of the nsub plant steps into the delay line; the motor driver holds
//                 the target sent `lag` plant steps ago
#pragma once

#include "firmware.hpp"
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
    // command path (policy-step mode when policy_target != null)
    const T* policy_target = nullptr;  // B x NJ  what the policy commands
    T* last_target = nullptr;          // B x NJ  firmware state: last sent target
    T* ring = nullptr;                 // B x ring_len x NJ  delay line (sent targets, one per plant step)
    int* ring_idx = nullptr;           // B  write position
    const int* lag = nullptr;          // B  delay in plant steps, 0 .. ring_len-1
    int ring_len = 0;
    const T* push = nullptr;           // B x 3  world-frame trunk velocity change at the policy step (or null)
    const T* kp = nullptr;
    const T* kd = nullptr;
    // outputs
    T* out_nu = nullptr;
    T* out_i = nullptr;
    T* out_tau = nullptr;
    T* out_v = nullptr;
    T* out_fc = nullptr;
    uint8_t* out_active = nullptr;
    uint8_t* out_slip = nullptr;
    T* out_gyro = nullptr;
    T* out_acc = nullptr;
    T* out_target = nullptr;  // B x NJ  the target the driver held in the last plant step
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
    io.target = nullptr;  // set per plant step in policy-step mode
    io.kp = b.kp ? b.kp + r * NJ : nullptr;
    io.kd = b.kd ? b.kd + r * NJ : nullptr;
    io.out_nu = b.out_nu ? b.out_nu + r * NV : nullptr;
    io.out_i = b.out_i ? b.out_i + r * NJ : nullptr;
    io.out_tau = b.out_tau ? b.out_tau + r * NJ : nullptr;
    io.out_v = b.out_v ? b.out_v + r * NJ : nullptr;
    io.out_fc = b.out_fc ? b.out_fc + r * nc * 3 : nullptr;
    io.out_active = b.out_active ? b.out_active + r * nc : nullptr;
    io.out_slip = b.out_slip ? b.out_slip + r * nc : nullptr;
    io.out_gyro = b.out_gyro ? b.out_gyro + r * 3 : nullptr;
    io.out_acc = b.out_acc ? b.out_acc + r * 3 : nullptr;
    return io;
}

// everything one robot does in one call: nsub plant steps, through the command path in policy-step mode
template <class T>
PH_HD void run_robot(const ModelData<T>& m, const BatchT<T>& b, int r, Work<T>& w) {
    StepIO<T> io = make_io(b, r, m.ncontact);
    if (!b.policy_target) {
        for (int s = 0; s < b.nsub; ++s) plant_step(m, io, w);
        return;
    }
    // the state at the policy step (and an optional push: a change of the trunk's world velocity)
    kinematics_and_velocity(m, io.q, io.p, w);
    if (b.push && !m.fixed_base) {
        const T* dv = b.push + r * 3;
        if (dv[0] != T(0) || dv[1] != T(0) || dv[2] != T(0)) {
            const T* R0 = w.f.R[0];  // v_B = R^T r_dot: the body-frame part of the world change
            for (int i = 0; i < 3; ++i) w.nu[3 + i] += R0[0 * 3 + i] * dv[0] + R0[1 * 3 + i] * dv[1] + R0[2 * 3 + i] * dv[2];
            for (int a = 0; a < NV; ++a) {
                T s = 0;
                for (int c = 0; c < NV; ++c) s += w.M[a * NV + c] * w.nu[c];
                io.p[a] = s;
            }
        }
    }
    // the firmware filter once per policy step, on the measured joint state
    T sent[NJ];
    safety_filter(m, b.policy_target + r * NJ, b.last_target + r * NJ, io.q + 7, w.nu + NB, io.kp, io.kd, sent);
    // every plant step: into the delay line; the driver holds the target sent `lag` steps ago
    T* ring = b.ring + static_cast<long>(r) * b.ring_len * NJ;
    int idx = b.ring_idx[r];
    int lag = b.lag[r];
    lag = lag < 0 ? 0 : (lag > b.ring_len - 1 ? b.ring_len - 1 : lag);
    for (int s = 0; s < b.nsub; ++s) {
        for (int k = 0; k < NJ; ++k) ring[idx * NJ + k] = sent[k];
        int ri = idx - lag;
        if (ri < 0) ri += b.ring_len;
        io.target = ring + ri * NJ;
        if (b.out_target && s == b.nsub - 1)
            for (int k = 0; k < NJ; ++k) b.out_target[r * NJ + k] = io.target[k];
        plant_step(m, io, w);
        idx = idx + 1 == b.ring_len ? 0 : idx + 1;
    }
    b.ring_idx[r] = idx;
}

template <class T> void run_cpu(const ModelData<T>& m, const BatchT<T>& b);
#ifdef PH_K1_WITH_CUDA
// md_device: the ModelData block in GPU memory; all batch pointers are GPU pointers; `stream` is a cudaStream_t
template <class T> void run_cuda(const ModelData<T>* md_device, int ncontact, const BatchT<T>& b, void* stream);
#endif

}  // namespace ph
