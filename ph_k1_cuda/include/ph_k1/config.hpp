// Sizes, the host/device macro and scalar math that compiles with g++ and nvcc alike.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define PH_HD __host__ __device__ __forceinline__
#else
#define PH_HD inline
#endif

namespace ph {

constexpr int NB = 6;             // trunk dofs (omega_B, v_B)
constexpr int NJ = 22;            // hinges
constexpr int NV = NB + NJ;       // generalised velocity nu = (omega_B, v_B, theta_dot)
constexpr int NQ = 7 + NJ;        // q = (r, quaternion w x y z, theta), MuJoCo's qpos layout
constexpr int MAX_BODIES = 32;    // the K1 has 25
constexpr int MAX_CONTACTS = 192; // the K1 has 128 (8 sole points + 8 box corners per other body)
constexpr int LEDGER = 6;         // supplied, mech_input, copper, gearbox, stops, contact  (plant.py Ledger)

// float/double overloads that exist on the host (C library) and on the device (CUDA math)
PH_HD float p_sqrt(float x) { return sqrtf(x); }
PH_HD double p_sqrt(double x) { return sqrt(x); }
PH_HD float p_sin(float x) { return sinf(x); }
PH_HD double p_sin(double x) { return sin(x); }
PH_HD float p_cos(float x) { return cosf(x); }
PH_HD double p_cos(double x) { return cos(x); }
PH_HD float p_abs(float x) { return fabsf(x); }
PH_HD double p_abs(double x) { return fabs(x); }

template <class T> PH_HD T p_max(T a, T b) { return a > b ? a : b; }
template <class T> PH_HD T p_min(T a, T b) { return a < b ? a : b; }
template <class T> PH_HD T p_clamp(T x, T lo, T hi) { return p_max(p_min(x, hi), lo); }

}  // namespace ph
