// 3-D vectors, row-major 3x3 matrices, quaternions (w x y z) and 6-D spatial vectors.
// Conventions as ph_k1/spatial.py (Featherstone, world frame, about the world origin O):
//   twist  V = (omega, v_O),   wrench W = (n_O, f),   W . V = n.omega + f.v
#pragma once

#include "config.hpp"

namespace ph {

template <class T> PH_HD T dot3(const T* a, const T* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

template <class T> PH_HD void cross3(const T* a, const T* b, T* o) {
    const T x = a[1] * b[2] - a[2] * b[1];
    const T y = a[2] * b[0] - a[0] * b[2];
    const T z = a[0] * b[1] - a[1] * b[0];
    o[0] = x; o[1] = y; o[2] = z;
}

// o = R v
template <class T> PH_HD void mv3(const T* R, const T* v, T* o) {
    const T x = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    const T y = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    const T z = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
    o[0] = x; o[1] = y; o[2] = z;
}

// C = A B
template <class T> PH_HD void mm3(const T* A, const T* B, T* C) {
    T out[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c] + A[r * 3 + 1] * B[1 * 3 + c] + A[r * 3 + 2] * B[2 * 3 + c];
    for (int i = 0; i < 9; ++i) C[i] = out[i];
}

// C = A B^T
template <class T> PH_HD void mmt3(const T* A, const T* B, T* C) {
    T out[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out[r * 3 + c] = A[r * 3 + 0] * B[c * 3 + 0] + A[r * 3 + 1] * B[c * 3 + 1] + A[r * 3 + 2] * B[c * 3 + 2];
    for (int i = 0; i < 9; ++i) C[i] = out[i];
}

// Rodrigues for a unit axis a: I + sin(t) [a]x + (1 - cos(t)) [a]x^2   (dynamics.py forward_kinematics)
template <class T> PH_HD void axis_angle(const T* a, T t, T* R) {
    const T K[9] = {0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0};
    T K2[9];
    mm3(K, K, K2);
    const T s = p_sin(t), c1 = T(1) - p_cos(t);
    for (int i = 0; i < 9; ++i) R[i] = s * K[i] + c1 * K2[i];
    R[0] += T(1); R[4] += T(1); R[8] += T(1);
}

template <class T> PH_HD void quat_to_mat(const T* q, T* R) {
    const T w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z);     R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y);     R[7] = 2 * (y * z + w * x);     R[8] = 1 - 2 * (x * x + y * y);
}

// Hamilton product o = a (x) b
template <class T> PH_HD void quat_mul(const T* a, const T* b, T* o) {
    const T w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    const T x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    const T y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    const T z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
    o[0] = w; o[1] = x; o[2] = y; o[3] = z;
}

// exponential map of a rotation vector (spatial.py quat_exp, same small-angle limit)
template <class T> PH_HD void quat_exp(const T* rv, T* o) {
    const T angle = p_sqrt(dot3(rv, rv));
    const T half = T(0.5) * angle;
    const T k = angle > T(1e-12) ? p_sin(half) / angle : T(0.5) - angle * angle / T(48);
    o[0] = p_cos(half);
    o[1] = k * rv[0]; o[2] = k * rv[1]; o[3] = k * rv[2];
}

template <class T> PH_HD T dot6(const T* a, const T* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4] + a[5] * b[5];
}

// a x_m b for twists: (w1 x w2, w1 x v2 + v1 x w2)
template <class T> PH_HD void cross_motion(const T* a, const T* b, T* o) {
    T t1[3], t2[3];
    cross3(a, b, o);
    cross3(a, b + 3, t1);
    cross3(a + 3, b, t2);
    o[3] = t1[0] + t2[0]; o[4] = t1[1] + t2[1]; o[5] = t1[2] + t2[2];
}

// Spatial inertia about O in the 13-number form (mass m, first moment m c, 3x3 inertia about O).
// It is additive, so composite (subtree) inertias are plain sums.  Applied to a twist (w, v):
//   I V = (I_O w + (m c) x v,  m v - (m c) x w)        (= spatial.py spatial_inertia @ V)
template <class T>
struct Inertia {
    T m;
    T mc[3];
    T IO[9];
};

template <class T> PH_HD void inertia_zero(Inertia<T>& I) {
    I.m = 0;
    for (int i = 0; i < 3; ++i) I.mc[i] = 0;
    for (int i = 0; i < 9; ++i) I.IO[i] = 0;
}

template <class T> PH_HD void inertia_add(Inertia<T>& a, const Inertia<T>& b) {
    a.m += b.m;
    for (int i = 0; i < 3; ++i) a.mc[i] += b.mc[i];
    for (int i = 0; i < 9; ++i) a.IO[i] += b.IO[i];
}

template <class T> PH_HD void inertia_apply(const Inertia<T>& I, const T* V, T* out) {
    T a[3], b[3];
    mv3(I.IO, V, a);
    cross3(I.mc, V + 3, b);
    T c[3];
    cross3(I.mc, V, c);
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
    out[3] = I.m * V[3] - c[0]; out[4] = I.m * V[4] - c[1]; out[5] = I.m * V[5] - c[2];
}

}  // namespace ph
