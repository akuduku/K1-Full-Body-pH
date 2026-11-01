// Rigid-body part of the K1 Hamiltonian for ONE robot (ph_k1/dynamics.py, same formulas):
// forward kinematics, the world motion axes S of the 28 dofs, the mass matrix M(q), body twists,
// spatial momenta, the analytic dT/dtheta and point velocities / Jacobian rows.
//
// The algorithms are the cheap tree versions of the Python ones, with the same results:
//   M:        composite rigid bodies.  M[e][d] = S_e . (Ic[body(d)] S_d) when dof e acts on an ancestor
//             (or the same body) of dof d, else 0; Ic = inertia of the subtree, about O.  This is
//             sum_b J_b^T I_b J_b + diag(armature) of dynamics.py, term by term.
//   forces:   generalised force of body wrenches W_b = sum over dofs S_d . Wc[body(d)], Wc = subtree sum.
#pragma once

#include "model_data.hpp"
#include "spatial.hpp"

namespace ph {

template <class T>
struct Frames {
    T R[MAX_BODIES][9];  // body orientation (world)
    T P[MAX_BODIES][3];  // body origin (world)
    T S[NV][6];          // world motion axis of every dof
    int dof_body[NV];    // body the dof moves
    Inertia<T> body[MAX_BODIES];  // per-body spatial inertia about O
    Inertia<T> comp[MAX_BODIES];  // subtree (composite) inertia about O
};

// dofs that move body b: the 6 trunk dofs, one hinge, or none (welded)
template <class T>
PH_HD int body_dofs(const ModelData<T>& m, int b, int* out) {
    const int j = m.body_joint[b];
    if (j == -2) {
        for (int i = 0; i < NB; ++i) out[i] = i;
        return NB;
    }
    if (j >= 0) {
        out[0] = NB + j;
        return 1;
    }
    return 0;
}

// q (NQ) -> frames, motion axes, per-body and composite inertias
template <class T>
PH_HD void forward_kinematics(const ModelData<T>& m, const T* q, Frames<T>& f) {
    const T* r = q;
    const T* Q = q + 3;
    const T* th = q + 7;
    for (int b = 0; b < m.nbody; ++b) {
        const int j = m.body_joint[b];
        if (j == -2) {  // the free trunk: pose = (r, Q)
            quat_to_mat(Q, f.R[b]);
            f.P[b][0] = r[0]; f.P[b][1] = r[1]; f.P[b][2] = r[2];
            continue;
        }
        const int pb = m.parent[b];
        T Rloc[9], Ploc[3];
        if (j >= 0) {  // hinge: R_loc = rot @ Rh, P_loc = pos + rot (jpos - Rh jpos)
            T Rh[9], t[3];
            axis_angle(m.jnt_axis[j], th[j], Rh);
            mm3(m.body_rot[b], Rh, Rloc);
            mv3(Rh, m.jnt_pos[j], t);
            for (int i = 0; i < 3; ++i) t[i] = m.jnt_pos[j][i] - t[i];
            mv3(m.body_rot[b], t, Ploc);
            for (int i = 0; i < 3; ++i) Ploc[i] += m.body_pos[b][i];
        } else {  // welded
            for (int i = 0; i < 9; ++i) Rloc[i] = m.body_rot[b][i];
            for (int i = 0; i < 3; ++i) Ploc[i] = m.body_pos[b][i];
        }
        mm3(f.R[pb], Rloc, f.R[b]);
        T t[3];
        mv3(f.R[pb], Ploc, t);
        for (int i = 0; i < 3; ++i) f.P[b][i] = f.P[pb][i] + t[i];
    }

    // hinge axes S = (a, o x a): a = R_body axis, o = anchor in the world
    for (int k = 0; k < NJ; ++k) {
        const int b = m.jnt_body[k];
        T a[3], o[3];
        mv3(f.R[b], m.jnt_axis[k], a);
        mv3(f.R[b], m.jnt_pos[k], o);
        for (int i = 0; i < 3; ++i) o[i] += f.P[b][i];
        T* S = f.S[NB + k];
        S[0] = a[0]; S[1] = a[1]; S[2] = a[2];
        cross3(o, a, S + 3);
        f.dof_body[NB + k] = b;
    }
    // trunk columns: body twist (omega_B, v_B) -> world twist.  omega: (R e_i, r x R e_i); v: (0, R e_i)
    {
        const T* Rt = f.R[0];
        for (int i = 0; i < 3; ++i) {
            const T col[3] = {Rt[0 * 3 + i], Rt[1 * 3 + i], Rt[2 * 3 + i]};
            T* Sw = f.S[i];
            Sw[0] = col[0]; Sw[1] = col[1]; Sw[2] = col[2];
            cross3(r, col, Sw + 3);
            T* Sv = f.S[3 + i];
            Sv[0] = 0; Sv[1] = 0; Sv[2] = 0;
            Sv[3] = col[0]; Sv[4] = col[1]; Sv[5] = col[2];
            f.dof_body[i] = 0;
            f.dof_body[3 + i] = 0;
        }
    }

    // spatial inertia about O: m, m c, I_O = R I_body R^T + m (|c|^2 1 - c c^T)
    for (int b = 0; b < m.nbody; ++b) {
        Inertia<T>& I = f.body[b];
        T c[3];
        mv3(f.R[b], m.ipos[b], c);
        for (int i = 0; i < 3; ++i) c[i] += f.P[b][i];
        T RI[9];
        mm3(f.R[b], m.I_body[b], RI);
        mmt3(RI, f.R[b], I.IO);
        const T mb = m.mass[b];
        const T cc = dot3(c, c);
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s) I.IO[r * 3 + s] += mb * ((r == s ? cc : T(0)) - c[r] * c[s]);
        I.m = mb;
        for (int i = 0; i < 3; ++i) I.mc[i] = mb * c[i];
        f.comp[b] = I;
    }
    for (int b = m.nbody - 1; b > 0; --b) inertia_add(f.comp[m.parent[b]], f.comp[b]);
}

// M (NV x NV, full, row-major) by composite rigid bodies, + armature on the hinge diagonal
template <class T>
PH_HD void mass_matrix(const ModelData<T>& m, const Frames<T>& f, T* M) {
    for (int i = 0; i < NV * NV; ++i) M[i] = 0;
    int dofs[NB];
    for (int d = 0; d < NV; ++d) {
        const int bd = f.dof_body[d];
        T F[6];
        inertia_apply(f.comp[bd], f.S[d], F);
        for (int a = bd; a >= 0; a = m.parent[a]) {
            const int n = body_dofs(m, a, dofs);
            for (int t = 0; t < n; ++t) {
                const int e = dofs[t];
                const T v = dot6(f.S[e], F);
                M[e * NV + d] = v;
                M[d * NV + e] = v;
            }
        }
    }
    for (int k = 0; k < NJ; ++k) M[(NB + k) * NV + NB + k] += m.armature[k];
}

// world twists of all bodies, V_b = J_b nu (forward pass)
template <class T>
PH_HD void body_twists(const ModelData<T>& m, const Frames<T>& f, const T* nu, T (*V)[6]) {
    for (int b = 0; b < m.nbody; ++b) {
        const int j = m.body_joint[b];
        if (j == -2) {
            for (int i = 0; i < 6; ++i) {
                T s = 0;
                for (int d = 0; d < NB; ++d) s += f.S[d][i] * nu[d];
                V[b][i] = s;
            }
            continue;
        }
        const int pb = m.parent[b];
        for (int i = 0; i < 6; ++i) V[b][i] = V[pb][i];
        if (j >= 0)
            for (int i = 0; i < 6; ++i) V[b][i] += f.S[NB + j][i] * nu[NB + j];
    }
}

// dT/dtheta at fixed nu (= -dH_kin/dtheta at fixed p):  -H_k . (S_k x_m V_parent(k)),
// H_k = spatial momentum of the subtree after hinge k (backward pass).  `work` holds nbody x 6.
template <class T>
PH_HD void kinetic_gradient(const ModelData<T>& m, const Frames<T>& f, const T (*V)[6], T (*H)[6], T* out) {
    for (int b = 0; b < m.nbody; ++b) inertia_apply(f.body[b], V[b], H[b]);
    for (int b = m.nbody - 1; b > 0; --b)
        for (int i = 0; i < 6; ++i) H[m.parent[b]][i] += H[b][i];
    for (int k = 0; k < NJ; ++k) {
        T c[6];
        cross_motion(f.S[NB + k], V[m.jnt_parent_body[k]], c);
        out[k] = -dot6(H[m.jnt_body[k]], c);
    }
}

// sum over dofs S_d . Wc[body(d)] for body wrenches W (destroys W: it becomes the subtree sums)
template <class T>
PH_HD void add_generalized_force(const ModelData<T>& m, const Frames<T>& f, T (*W)[6], T* F) {
    for (int b = m.nbody - 1; b > 0; --b)
        for (int i = 0; i < 6; ++i) W[m.parent[b]][i] += W[b][i];
    for (int d = 0; d < NV; ++d) F[d] += dot6(f.S[d], W[f.dof_body[d]]);
}

// generalised gravity force: subtree wrench of gravity = ((m c)_sub x g, m_sub g)
template <class T>
PH_HD void add_gravity(const ModelData<T>& m, const Frames<T>& f, T* F) {
    for (int d = 0; d < NV; ++d) {
        const Inertia<T>& I = f.comp[f.dof_body[d]];
        T W[6];
        cross3(I.mc, m.g, W);
        W[3] = I.m * m.g[0]; W[4] = I.m * m.g[1]; W[5] = I.m * m.g[2];
        F[d] += dot6(f.S[d], W);
    }
}

// velocity of a world point x fixed to a body with twist V:  v_O + omega x x
template <class T>
PH_HD void point_velocity(const T* V, const T* x, T* v) {
    cross3(V, x, v);
    v[0] += V[3]; v[1] += V[4]; v[2] += V[5];
}

// The non-zero columns of the point Jacobian of x on body b (dynamics.py point_jacobian):
// column d = S_d.v + S_d.w x x for every dof d on the path trunk -> b.  Returns their count.
template <class T>
PH_HD int point_jacobian(const ModelData<T>& m, const Frames<T>& f, int b, const T* x, int* idx, T (*col)[3]) {
    int n = 0, dofs[NB];
    for (int a = b; a >= 0; a = m.parent[a]) {
        const int k = body_dofs(m, a, dofs);
        for (int t = 0; t < k; ++t) {
            const int d = dofs[t];
            point_velocity(f.S[d], x, col[n]);
            idx[n++] = d;
        }
    }
    return n;
}

}  // namespace ph
