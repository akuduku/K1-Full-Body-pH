// Dense symmetric positive-definite solves for small fixed sizes (n <= NV), row-major with leading
// dimension LD.  Both systems of a plant step are SPD: the momentum block A = M + h D + h J^T K J and
// the motor block Ksys = diag(L + h R) + h^2 Ks (B^T A^-1 B) Ks.
#pragma once

#include "config.hpp"

namespace ph {

// In place: the lower triangle of A becomes L with A = L L^T.  Returns false if A is not SPD.
template <class T, int LD>
PH_HD bool cholesky(T* A, int n) {
    for (int j = 0; j < n; ++j) {
        T d = A[j * LD + j];
        for (int k = 0; k < j; ++k) d -= A[j * LD + k] * A[j * LD + k];
        if (!(d > T(0))) return false;
        d = p_sqrt(d);
        A[j * LD + j] = d;
        const T inv = T(1) / d;
        for (int i = j + 1; i < n; ++i) {
            T s = A[i * LD + j];
            for (int k = 0; k < j; ++k) s -= A[i * LD + k] * A[j * LD + k];
            A[i * LD + j] = s * inv;
        }
    }
    return true;
}

// x <- (L L^T)^-1 x
template <class T, int LD>
PH_HD void cholesky_solve(const T* L, int n, T* x) {
    for (int i = 0; i < n; ++i) {
        T s = x[i];
        for (int k = 0; k < i; ++k) s -= L[i * LD + k] * x[k];
        x[i] = s / L[i * LD + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        T s = x[i];
        for (int k = i + 1; k < n; ++k) s -= L[k * LD + i] * x[k];
        x[i] = s / L[i * LD + i];
    }
}

}  // namespace ph
