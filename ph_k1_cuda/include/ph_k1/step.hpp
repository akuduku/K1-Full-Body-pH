// One plant step h for ONE robot: ph_k1/plant.py K1PHPlant.step, term by term, plus the motor driver of the
// robot's command path (ph_k1/firmware.py motor_volts: PD on the delayed target -> volts).  Called by the CPU
// loop (one robot per OpenMP iteration) and the CUDA kernel (one robot per thread) alike.
//
//   explicit at x_n: gyroscopic, dT/dtheta, gravity, test torque, spring parts of contact and end stops
//   implicit:        dampers, contact/end-stop stiffness (linearised), motor gyrator, coil resistance,
//                    one linear system in (nu_{n+1}, i_{n+1}) via the Schur complement of the motor block
//   contact:         solve with every penetrating point sticking; if the Coulomb cone or the unilateral
//                    condition says otherwise, solve once more with the corrected modes (per robot)
#pragma once

#include "dynamics.hpp"
#include "linalg.hpp"

namespace ph {

// pointers to the rows of ONE robot (state is updated in place)
template <class T>
struct StepIO {
    T* q;            // NQ
    T* p;            // NV
    T* xi;           // ncontact x 2  tangential bristles
    T* lam;          // NJ coil fluxes
    T* ledger;       // LEDGER
    const T* volts;  // NJ, null = 0 V (ignored while the motor driver runs)
    const T* jtau;   // NJ test torque port, null = none
    const T* fext;   // 3 world-frame force on the trunk origin (e.g. a get-up assist), null = none
    const T* fgen;   // NV generalised force (e.g. J^T f of a push on any body point), null = none
    const T* rp;     // NRP this robot's parameter row (config.hpp RP_*), null = the ModelData values
    T* ball;         // NBALL this robot's ball (centre, linear velocity, angular velocity), null = no ball
    // motor driver; active when target != null (set per plant step by the command path in launch.hpp)
    const T* target;    // NJ the target the driver holds this step (already filtered and delayed)
    const T* kp;        // NJ
    const T* kd;        // NJ
    // outputs of this step, each may be null
    T* out_nu;          // NV  nu_{n+1}
    T* out_i;           // NJ  i_{n+1}
    T* out_tau;         // NJ  motor torque s k i_{n+1}
    T* out_v;           // NJ  volts applied
    T* out_fc;          // ncontact x 3 world contact force
    uint8_t* out_active;  // ncontact  normal contact on
    uint8_t* out_slip;    // ncontact  slipping
    T* out_gyro;        // 3  trunk angular velocity, trunk frame (the robot's gyro)
    T* out_acc;         // 3  specific force of the trunk origin, trunk frame (the robot's accelerometer)
};

template <class T>
struct Work {
    Frames<T> f;
    T M[NV * NV];
    T A[NV * NV];       // momentum block, then its Cholesky factor
    T Y[NV * NJ];       // A^-1 B (row = free dof, column = hinge)
    T K[NJ * NJ];       // motor block, then its Cholesky factor
    T V[MAX_BODIES][6]; // body twists
    T H[MAX_BODIES][6]; // momenta / wrench accumulator
    T nu[NV], nu0[NV], nu1[NV], F[NV], Fc[NV];
    T i_n[NJ], i1[NJ], v[NJ], Ks[NJ], diag[NJ];
    uint8_t stop_on[NJ];
    T x[MAX_CONTACTS][3];   // world contact points at x_n
    T delta[MAX_CONTACTS];  // penetration -z
    uint8_t cand[MAX_CONTACTS], normal[MAX_CONTACTS], stick[MAX_CONTACTS];
    T fslip[MAX_CONTACTS][2];
    // self-contact: sphere centres at x_n, and the overlapping pairs of this step for the ledger
    T sx[MAX_SPHERES][3];
    int self_pair[MAX_SELF_ACTIVE];
    uint8_t self_on[MAX_SELF_ACTIVE];
    int nself;
    // ball: the boxes it overlaps at x_n (normal robot -> ball, row = n . J_c), the ground, and the 3x3 block of its
    // linear velocity that implicit_solve eliminates
    int nball;
    uint8_t ball_on[MAX_BALL_BOXES];
    T ball_delta[MAX_BALL_BOXES], ball_n[MAX_BALL_BOXES][3], ball_row[MAX_BALL_BOXES][NV];
    T ball_x[MAX_BALL_BOXES][3];  // the boxes' contact points
    int ball_body[MAX_BALL_BOXES];
    T ball_dp[NV];                // friction's reaction on the robot's momentum
    T ball_dg;
    uint8_t ground_cand, ground_on;
    T ball_Ainv[9], ball_bb[3], ball_C[NV][3], ball_u1[3];
    // this robot's effective parameters (ModelData, changed by its parameter row): set by effective_params
    T k_n, c_n, k_t, c_t, mu;
    T damping[NJ], tau_max[NJ];
};

// ModelData values, changed by the robot's parameter row if it has one (config.hpp RP_*)
template <class T>
PH_HD void effective_params(const ModelData<T>& m, const StepIO<T>& io, Work<T>& w) {
    const T* r = io.rp;
    w.k_n = r ? m.k_n * r[RP_KN] : m.k_n;
    w.c_n = r ? m.c_n * r[RP_CN] : m.c_n;
    w.k_t = r ? m.k_t * r[RP_KT] : m.k_t;
    w.c_t = r ? m.c_t * r[RP_CT] : m.c_t;
    w.mu = r ? m.mu * r[RP_MU] : m.mu;
    for (int k = 0; k < NJ; ++k) {
        w.damping[k] = r ? m.damping[k] + r[RP_DAMP + k] : m.damping[k];
        w.tau_max[k] = r ? m.tau_max[k] * r[RP_TAU + k] : m.tau_max[k];
    }
}

template <class T> PH_HD T gyrator_scale(const ModelData<T>& m, const Work<T>& w, int k, T i) {
    const T ki = p_abs(m.mk[k] * i);
    return ki > w.tau_max[k] ? w.tau_max[k] / p_max(ki, T(1e-30)) : T(1);
}

// The motor driver (firmware.py motor_volts): PD on the target it holds, limited by the motor's torque
// envelope at this speed, and a one-step current loop matched to the plant's implicit electrical row.
// Clamp, slew and delay happened before (firmware.hpp, launch.hpp).
template <class T>
PH_HD void driver_volts(const ModelData<T>& m, const StepIO<T>& io, Work<T>& w) {
    const T* th = io.q + 7;
    for (int k = 0; k < NJ; ++k) {
        const T qd = w.nu[NB + k];
        const T i = w.i_n[k];
        T tau = io.kp[k] * (io.target[k] - th[k]) - io.kd[k] * qd;
        const T sgn = tau > T(0) ? T(1) : (tau < T(0) ? T(-1) : T(0));
        const T avail = p_max((m.v_max - m.mk[k] * qd * sgn) * m.mk[k] / m.mR[k], T(0));
        const T cap = p_min(avail, w.tau_max[k]);
        tau = p_max(p_min(tau, cap), -cap);
        const T i_star = tau / m.mk[k];
        const T s = gyrator_scale(m, w, k, i);
        const T v = m.mL[k] * (i_star - i) / m.h + m.mR[k] * i_star + s * m.mk[k] * qd;
        w.v[k] = p_clamp(v, -m.v_max, m.v_max);
    }
}

// Self-contact (ph_k1/self_contact.py).  The sphere centres at x_n -> w.sx
template <class T>
PH_HD void self_spheres(const ModelData<T>& m, Work<T>& w) {
    for (int s = 0; s < m.nsphere; ++s) {
        const int b = m.sphere_body[s];
        mv3(w.f.R[b], m.sphere_pos[s], w.sx[s]);
        for (int i = 0; i < 3; ++i) w.sx[s][i] += w.f.P[b][i];
    }
}

// pair p: false if apart, else its overlap, the normal (b -> a) and the force point (the middle of the overlap)
template <class T>
PH_HD bool self_pair(const ModelData<T>& m, const Work<T>& w, int p, T& delta, T* n, T* x) {
    const int a = m.pair_a[p], b = m.pair_b[p];
    const T dv[3] = {w.sx[a][0] - w.sx[b][0], w.sx[a][1] - w.sx[b][1], w.sx[a][2] - w.sx[b][2]};
    const T rr = m.sphere_r[a] + m.sphere_r[b];
    const T d2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
    if (d2 >= rr * rr) return false;
    const T d = p_sqrt(d2);
    delta = rr - d;
    const T id = T(1) / p_max(d, T(1e-9));
    const T rb = m.sphere_r[b] - T(0.5) * delta;
    for (int i = 0; i < 3; ++i) {
        n[i] = dv[i] * id;
        x[i] = w.sx[b][i] + n[i] * rb;
    }
    return true;
}

// n . (v_a - v_b) at the force point of pair p, with the body twists in w.V
template <class T>
PH_HD T self_normal_speed(const ModelData<T>& m, const Work<T>& w, int p, const T* n, const T* x) {
    T va[3], vb[3];
    point_velocity(w.V[m.sphere_body[m.pair_a[p]]], x, va);
    point_velocity(w.V[m.sphere_body[m.pair_b[p]]], x, vb);
    return (va[0] - vb[0]) * n[0] + (va[1] - vb[1]) * n[1] + (va[2] - vb[2]) * n[2];
}

// the overlapping pairs of this step -> w.self_pair, all on
template <class T>
PH_HD void self_collect(const ModelData<T>& m, Work<T>& w) {
    w.nself = 0;
    for (int p = 0; p < m.npair && w.nself < MAX_SELF_ACTIVE; ++p) {
        T delta, n[3], x[3];
        if (!self_pair(m, w, p, delta, n, x)) continue;
        w.self_pair[w.nself] = p;
        w.self_on[w.nself] = 1;
        ++w.nself;
    }
}

// row[NV] = n . (J_a - J_b) of pair p at its force point
template <class T>
PH_HD void self_row(const ModelData<T>& m, const Work<T>& w, int p, const T* n, const T* x, T* row) {
    for (int d = 0; d < NV; ++d) row[d] = 0;
    int idx[NV];
    T col[NV][3];
    int nz = point_jacobian(m, w.f, m.sphere_body[m.pair_a[p]], x, idx, col);
    for (int s = 0; s < nz; ++s) row[idx[s]] += n[0] * col[s][0] + n[1] * col[s][1] + n[2] * col[s][2];
    nz = point_jacobian(m, w.f, m.sphere_body[m.pair_b[p]], x, idx, col);
    for (int s = 0; s < nz; ++s) row[idx[s]] -= n[0] * col[s][0] + n[1] * col[s][1] + n[2] * col[s][2];
}

// Ball (ph_k1/ball.py).  Box j against the ball's centre xb: false if apart, else the overlap, the normal
// (robot -> ball) and the box's closest point x
template <class T>
PH_HD bool ball_pair(const ModelData<T>& m, const Work<T>& w, const T* xb, int j, T& delta, T* n, T* x) {
    const int b = m.box_body[j];
    const T* half = m.box_half[j];
    T Rw[9], cw[3], d[3], loc[3], cl[3], nl[3];
    mm3(w.f.R[b], m.box_rot[j], Rw);
    mv3(w.f.R[b], m.box_pos[j], cw);
    for (int i = 0; i < 3; ++i) {
        cw[i] += w.f.P[b][i];
        d[i] = xb[i] - cw[i];
    }
    T d2 = 0;
    for (int i = 0; i < 3; ++i) {
        loc[i] = Rw[0 * 3 + i] * d[0] + Rw[1 * 3 + i] * d[1] + Rw[2 * 3 + i] * d[2];  // Rw^T d
        cl[i] = p_clamp(loc[i], -half[i], half[i]);
        nl[i] = loc[i] - cl[i];
        d2 += nl[i] * nl[i];
    }
    const T dist = p_sqrt(d2);
    if (dist < T(1e-9)) {  // the centre is inside the box: out through the nearest face
        int ax = 0;
        T room = half[0] - p_abs(loc[0]);
        for (int i = 1; i < 3; ++i) {
            const T r = half[i] - p_abs(loc[i]);
            if (r < room) {
                room = r;
                ax = i;
            }
        }
        const T sg = loc[ax] >= T(0) ? T(1) : T(-1);
        for (int i = 0; i < 3; ++i) {
            nl[i] = 0;
            cl[i] = loc[i];
        }
        nl[ax] = sg;
        cl[ax] = sg * half[ax];
        delta = m.ball_r + room;
    } else {
        if (dist >= m.ball_r) return false;
        for (int i = 0; i < 3; ++i) nl[i] /= dist;
        delta = m.ball_r - dist;
    }
    mv3(Rw, nl, n);
    mv3(Rw, cl, x);
    for (int i = 0; i < 3; ++i) x[i] += cw[i];
    return true;
}

// the boxes the ball overlaps at x_n and its ground overlap -> w.ball_*, all on
template <class T>
PH_HD void ball_collect(const ModelData<T>& m, const StepIO<T>& io, Work<T>& w) {
    w.nball = 0;
    for (int j = 0; j < m.nbox && w.nball < MAX_BALL_BOXES; ++j) {
        T delta = 0, n[3], x[3];
        if (!ball_pair(m, w, io.ball, j, delta, n, x)) continue;
        const int k = w.nball++;
        w.ball_on[k] = 1;
        w.ball_delta[k] = delta;
        for (int i = 0; i < 3; ++i) {
            w.ball_n[k][i] = n[i];
            w.ball_x[k][i] = x[i];
        }
        w.ball_body[k] = m.box_body[j];
        for (int d = 0; d < NV; ++d) w.ball_row[k][d] = 0;
        int idx[NV];
        T col[NV][3];
        const int nz = point_jacobian(m, w.f, m.box_body[j], x, idx, col);
        for (int s = 0; s < nz; ++s) w.ball_row[k][idx[s]] += n[0] * col[s][0] + n[1] * col[s][1] + n[2] * col[s][2];
    }
    w.ball_dg = p_max(m.ball_r - io.ball[2], T(0));
    w.ground_cand = w.ball_dg > T(0);
    w.ground_on = w.ground_cand;
}

// inverse of a 3x3 matrix (row-major)
template <class T>
PH_HD void inv3(const T* a, T* o) {
    const T c00 = a[4] * a[8] - a[5] * a[7], c01 = a[5] * a[6] - a[3] * a[8], c02 = a[3] * a[7] - a[4] * a[6];
    const T id = T(1) / (a[0] * c00 + a[1] * c01 + a[2] * c02);
    o[0] = c00 * id;
    o[1] = (a[2] * a[7] - a[1] * a[8]) * id;
    o[2] = (a[1] * a[5] - a[2] * a[4]) * id;
    o[3] = c01 * id;
    o[4] = (a[0] * a[8] - a[2] * a[6]) * id;
    o[5] = (a[2] * a[3] - a[0] * a[5]) * id;
    o[6] = c02 * id;
    o[7] = (a[1] * a[6] - a[0] * a[7]) * id;
    o[8] = (a[0] * a[4] - a[1] * a[3]) * id;
}

// The implicit solve for the given contact modes -> w.nu1, w.i1   (plant.py step.solve)
template <class T>
PH_HD void implicit_solve(const ModelData<T>& m, const StepIO<T>& io, Work<T>& w, int f0, int n, bool use_contact) {
    const T h = m.h;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) w.A[a * NV + b] = w.M[(a + f0) * NV + (b + f0)];
    for (int k = 0; k < NJ; ++k) w.A[(NB + k - f0) * NV + (NB + k - f0)] += h * w.diag[k];
    for (int d = 0; d < NV; ++d) w.Fc[d] = w.F[d];

    if (m.self_enabled) {  // limbs against limbs: spring at x_n, h (c + h k) J^T J in the momentum block
        const T cs_eff = m.self_c + h * m.self_k;
        for (int i = 0; i < w.nself; ++i) {
            if (!w.self_on[i]) continue;
            T delta = 0, nv[3], x[3], row[NV];
            self_pair(m, w, w.self_pair[i], delta, nv, x);
            self_row(m, w, w.self_pair[i], nv, x, row);
            const T fs = m.self_k * delta;
            for (int d = 0; d < NV; ++d) w.Fc[d] += fs * row[d];
            for (int a = 0; a < n; ++a) {
                const T ga = h * cs_eff * row[a + f0];
                if (ga == T(0)) continue;
                for (int b = 0; b < n; ++b) w.A[a * NV + b] += ga * row[b + f0];
            }
        }
    }

    if (io.ball) {  // the ball's 3 linear velocities, eliminated from the system (Schur complement, ph_k1/ball.py)
        const T cb_eff = m.ball_c + h * m.ball_k, cg_eff = m.ball_cg + h * m.ball_kg;
        T Abb[9] = {m.ball_m, 0, 0, 0, m.ball_m, 0, 0, 0, m.ball_m};
        const T* u0 = io.ball + 3;
        for (int i = 0; i < 3; ++i) w.ball_bb[i] = m.ball_m * u0[i] + (m.gravity ? h * m.ball_m * m.g[i] : T(0));
        for (int d = 0; d < NV; ++d)
            for (int i = 0; i < 3; ++i) w.ball_C[d][i] = 0;
        bool touching = false;
        for (int k = 0; k < w.nball; ++k) {
            if (!w.ball_on[k]) continue;
            touching = true;
            const T* row = w.ball_row[k];
            const T* nv = w.ball_n[k];
            const T fs = m.ball_k * w.ball_delta[k];
            for (int d = 0; d < NV; ++d) w.Fc[d] -= fs * row[d];
            for (int a = 0; a < n; ++a) {
                const T ga = h * cb_eff * row[a + f0];
                if (ga == T(0)) continue;
                for (int b = 0; b < n; ++b) w.A[a * NV + b] += ga * row[b + f0];
            }
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) Abb[i * 3 + j] += h * cb_eff * nv[i] * nv[j];
                w.ball_bb[i] += h * fs * nv[i];
            }
            for (int d = 0; d < NV; ++d)
                for (int i = 0; i < 3; ++i) w.ball_C[d][i] += h * cb_eff * row[d] * nv[i];
        }
        if (w.ground_on) {
            Abb[8] += h * cg_eff;
            w.ball_bb[2] += h * m.ball_kg * w.ball_dg;
        }
        inv3(Abb, w.ball_Ainv);
        if (touching) {
            for (int a = 0; a < n; ++a) {
                const T* ca = w.ball_C[a + f0];
                T CA[3];
                for (int j = 0; j < 3; ++j)
                    CA[j] = ca[0] * w.ball_Ainv[0 * 3 + j] + ca[1] * w.ball_Ainv[1 * 3 + j] + ca[2] * w.ball_Ainv[2 * 3 + j];
                if (CA[0] == T(0) && CA[1] == T(0) && CA[2] == T(0)) continue;
                for (int b = 0; b < n; ++b) {
                    const T* cb = w.ball_C[b + f0];
                    w.A[a * NV + b] -= CA[0] * cb[0] + CA[1] * cb[1] + CA[2] * cb[2];
                }
            }
            T Ab[3];
            mv3(w.ball_Ainv, w.ball_bb, Ab);
            for (int d = 0; d < NV; ++d)
                w.Fc[d] += (w.ball_C[d][0] * Ab[0] + w.ball_C[d][1] * Ab[1] + w.ball_C[d][2] * Ab[2]) / h;
        }
    }

    if (use_contact) {
        const T cn_eff = w.c_n + h * w.k_n, ct_eff = w.c_t + h * w.k_t;
        for (int b = 0; b < m.nbody; ++b)
            for (int i = 0; i < 6; ++i) w.H[b][i] = 0;
        for (int c = 0; c < m.ncontact; ++c) {
            if (!w.normal[c] && !w.stick[c]) continue;
            const int b = m.contact_body[c];
            int idx[NV];
            T col[NV][3];
            const int nz = point_jacobian(m, w.f, b, w.x[c], idx, col);
            // h J^T diag(w) J on the free block, only over the non-zero columns
            for (int r = 0; r < 3; ++r) {
                const T wr = r == 2 ? (w.normal[c] ? cn_eff : T(0)) : (w.stick[c] ? ct_eff : T(0));
                if (wr == T(0)) continue;
                for (int s = 0; s < nz; ++s) {
                    const int a = idx[s] - f0;
                    if (a < 0) continue;
                    const T ga = h * wr * col[s][r];
                    for (int t = 0; t < nz; ++t) {
                        const int bb = idx[t] - f0;
                        if (bb < 0) continue;
                        w.A[a * NV + bb] += ga * col[t][r];
                    }
                }
            }
            // spring force of this mode: f0 = (-k_t xi stick + f_slip, k_n delta normal), as a body wrench
            T fz = w.normal[c] ? w.k_n * w.delta[c] : T(0);
            T fx = (w.stick[c] ? -w.k_t * io.xi[2 * c + 0] : T(0)) + w.fslip[c][0];
            T fy = (w.stick[c] ? -w.k_t * io.xi[2 * c + 1] : T(0)) + w.fslip[c][1];
            const T fv[3] = {fx, fy, fz};
            T mom[3];
            cross3(w.x[c], fv, mom);
            for (int i = 0; i < 3; ++i) {
                w.H[b][i] += mom[i];
                w.H[b][3 + i] += fv[i];
            }
        }
        add_generalized_force(m, w.f, w.H, w.Fc);
    }

    // nu0 = A^-1 (p + h Ftot),  Y = A^-1 B
    cholesky<T, NV>(w.A, n);
    for (int a = 0; a < n; ++a) w.nu0[a] = io.p[a + f0] + h * w.Fc[a + f0];
    cholesky_solve<T, NV>(w.A, n, w.nu0);
    for (int j = 0; j < NJ; ++j) {
        T e[NV];
        for (int a = 0; a < n; ++a) e[a] = 0;
        e[NB + j - f0] = 1;
        cholesky_solve<T, NV>(w.A, n, e);
        for (int a = 0; a < n; ++a) w.Y[a * NJ + j] = e[a];
    }
    // motor block: Ksys i1 = L i_n + h v - h Ks nu0[hinges]
    for (int i = 0; i < NJ; ++i)
        for (int j = 0; j < NJ; ++j)
            w.K[i * NJ + j] = h * h * w.Ks[i] * w.Y[(NB + i - f0) * NJ + j] * w.Ks[j] + (i == j ? m.mL[i] + h * m.mR[i] : T(0));
    for (int i = 0; i < NJ; ++i) w.i1[i] = m.mL[i] * w.i_n[i] + h * w.v[i] - h * w.Ks[i] * w.nu0[NB + i - f0];
    cholesky<T, NJ>(w.K, NJ);
    cholesky_solve<T, NJ>(w.K, NJ, w.i1);
    // nu1 = nu0 + h Y (Ks i1)
    for (int d = 0; d < NV; ++d) w.nu1[d] = 0;
    for (int a = 0; a < n; ++a) {
        T s = w.nu0[a];
        for (int j = 0; j < NJ; ++j) s += h * w.Y[a * NJ + j] * w.Ks[j] * w.i1[j];
        w.nu1[a + f0] = s;
    }
    if (io.ball) {  // u1 = A_bb^-1 (b_b + C^T nu1)
        T r[3] = {w.ball_bb[0], w.ball_bb[1], w.ball_bb[2]};
        for (int d = 0; d < NV; ++d)
            for (int i = 0; i < 3; ++i) r[i] += w.ball_C[d][i] * w.nu1[d];
        mv3(w.ball_Ainv, r, w.ball_u1);
    }
}

// kinematics, M(q) and nu = M^-1 p on the free dofs (zeros on a welded trunk) -> w.f, w.M, w.nu
template <class T>
PH_HD void kinematics_and_velocity(const ModelData<T>& m, const T* q, const T* p, Work<T>& w) {
    const int f0 = m.fixed_base ? NB : 0;
    const int n = NV - f0;
    forward_kinematics(m, q, w.f);
    mass_matrix(m, w.f, w.M);
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) w.A[a * NV + b] = w.M[(a + f0) * NV + (b + f0)];
    cholesky<T, NV>(w.A, n);
    for (int d = 0; d < NV; ++d) w.nu[d] = 0;
    for (int a = 0; a < n; ++a) w.nu[a + f0] = p[a + f0];
    cholesky_solve<T, NV>(w.A, n, w.nu + f0);
}

template <class T>
PH_HD void plant_step(const ModelData<T>& m, const StepIO<T>& io, Work<T>& w) {
    const T h = m.h;
    const int f0 = m.fixed_base ? NB : 0;
    const int n = NV - f0;
    const int nc = m.ncontact;
    T* q = io.q;
    T* th = q + 7;

    effective_params(m, io, w);
    // kinematics, M and nu at x_n
    kinematics_and_velocity(m, q, io.p, w);
    for (int k = 0; k < NJ; ++k) w.i_n[k] = io.lam[k] / m.mL[k];

    // input voltage
    if (io.target) {
        driver_volts(m, io, w);
    } else {
        for (int k = 0; k < NJ; ++k) w.v[k] = io.volts ? io.volts[k] : T(0);
    }

    // explicit, configuration-dependent part of p_dot
    for (int d = 0; d < NV; ++d) w.F[d] = 0;
    if (!m.fixed_base) {  // -nu_B x* p_B
        const T* pn = io.p;
        const T* pf = io.p + 3;
        const T* om = w.nu;
        const T* vb = w.nu + 3;
        T a[3], b[3], c[3];
        cross3(pn, om, a);
        cross3(pf, vb, b);
        cross3(pf, om, c);
        for (int i = 0; i < 3; ++i) {
            w.F[i] += a[i] + b[i];
            w.F[3 + i] += c[i];
        }
    }
    body_twists(m, w.f, w.nu, w.V);
    {
        T g[NJ];
        kinetic_gradient(m, w.f, w.V, w.H, g);
        for (int k = 0; k < NJ; ++k) w.F[NB + k] += g[k];
    }
    if (m.gravity) add_gravity(m, w.f, w.F);
    if (io.jtau)
        for (int k = 0; k < NJ; ++k) w.F[NB + k] += io.jtau[k];
    // a world force f on the trunk origin: generalised force (0, R^T f) on (omega_B, v_B); its power f.v
    T f_body[3] = {T(0), T(0), T(0)};
    const bool use_fext = io.fext && !m.fixed_base;
    if (use_fext) {
        const T* R0 = w.f.R[0];
        for (int i = 0; i < 3; ++i) {
            f_body[i] = R0[0 * 3 + i] * io.fext[0] + R0[1 * 3 + i] * io.fext[1] + R0[2 * 3 + i] * io.fext[2];
            w.F[3 + i] += f_body[i];
        }
    }
    if (io.fgen)
        for (int d = 0; d < NV; ++d) w.F[d] += io.fgen[d];
    w.nself = 0;
    if (m.self_enabled) {  // limbs against limbs: the overlapping pairs at x_n
        self_spheres(m, w);
        self_collect(m, w);
    }
    w.nball = 0;
    if (io.ball) ball_collect(m, io, w);

    // hinge diagonal: gearbox damping + end stops
    for (int k = 0; k < NJ; ++k) {
        w.diag[k] = w.damping[k];
        w.stop_on[k] = 0;
        if (m.limits_enabled) {
            const T over = p_max(th[k] - m.stop_high[k], T(0));
            const T under = p_max(m.stop_low[k] - th[k], T(0));
            w.stop_on[k] = (over > T(0)) || (under > T(0));
            w.F[NB + k] += m.lim_k * (under - over);
            if (w.stop_on[k]) w.diag[k] += m.lim_c + h * m.lim_k;
        }
    }

    // contact candidates
    const bool use_contact = m.contact_enabled && !m.fixed_base;
    bool any_cand = false;
    for (int c = 0; c < nc; ++c) {
        const int b = m.contact_body[c];
        mv3(w.f.R[b], m.contact_local[c], w.x[c]);
        for (int i = 0; i < 3; ++i) w.x[c][i] += w.f.P[b][i];
        w.delta[c] = -w.x[c][2];
        w.cand[c] = use_contact && w.delta[c] > T(0);
        any_cand = any_cand || w.cand[c];
        w.normal[c] = w.cand[c];
        w.stick[c] = w.cand[c];
        w.fslip[c][0] = 0;
        w.fslip[c][1] = 0;
    }

    for (int k = 0; k < NJ; ++k) w.Ks[k] = m.motors_connected ? gyrator_scale(m, w, k, w.i_n[k]) * m.mk[k] : T(0);

    implicit_solve(m, io, w, f0, n, use_contact);

    const T cn_eff = w.c_n + h * w.k_n, ct_eff = w.c_t + h * w.k_t;
    bool changed = false;
    if (use_contact && any_cand) {  // Coulomb cone and unilateral check, one corrected solve if needed
        body_twists(m, w.f, w.nu1, w.V);
        for (int c = 0; c < nc; ++c) {
            uint8_t n2 = 0, s2 = 0;
            T fs0 = 0, fs1 = 0;
            if (w.cand[c]) {
                T vc[3];
                point_velocity(w.V[m.contact_body[c]], w.x[c], vc);
                const T fz = w.k_n * w.delta[c] - cn_eff * vc[2];
                const T ft0 = -w.k_t * io.xi[2 * c + 0] - ct_eff * vc[0];
                const T ft1 = -w.k_t * io.xi[2 * c + 1] - ct_eff * vc[1];
                n2 = fz > T(0);
                const T fz_c = n2 ? p_max(fz, T(0)) : T(0);
                const T ftn = p_sqrt(ft0 * ft0 + ft1 * ft1);
                s2 = n2 && (ftn <= w.mu * fz_c);
                if (n2 && !s2) {
                    const T sc = w.mu * fz_c / p_max(ftn, T(1e-12));
                    fs0 = sc * ft0;
                    fs1 = sc * ft1;
                }
            }
            changed = changed || (n2 != w.normal[c]) || (s2 != w.stick[c]);
            w.normal[c] = n2;  // kept only if `changed` (otherwise equal anyway: f_slip is then 0)
            w.stick[c] = s2;
            w.fslip[c][0] = fs0;
            w.fslip[c][1] = fs1;
        }
    }
    const T cs_eff = m.self_c + h * m.self_k;
    if (w.nself > 0) {  // unilateral: the elements that would pull let go
        body_twists(m, w.f, w.nu1, w.V);
        for (int i = 0; i < w.nself; ++i) {
            if (!w.self_on[i]) continue;
            const int p = w.self_pair[i];
            T delta = 0, nv[3], x[3];
            self_pair(m, w, p, delta, nv, x);
            if (m.self_k * delta - cs_eff * self_normal_speed(m, w, p, nv, x) <= T(0)) {
                w.self_on[i] = 0;
                changed = true;
            }
        }
    }
    const T cb_eff = m.ball_c + h * m.ball_k, cg_eff = m.ball_cg + h * m.ball_kg;
    if (io.ball) {  // unilateral: neither a box nor the ground pulls the ball
        for (int k = 0; k < w.nball; ++k) {
            if (!w.ball_on[k]) continue;
            T rv = 0;
            for (int d = 0; d < NV; ++d) rv += w.ball_row[k][d] * w.nu1[d];
            const T g = dot3(w.ball_n[k], w.ball_u1) - rv;
            if (m.ball_k * w.ball_delta[k] - cb_eff * g <= T(0)) {
                w.ball_on[k] = 0;
                changed = true;
            }
        }
        if (w.ground_on && m.ball_kg * w.ball_dg - cg_eff * w.ball_u1[2] <= T(0)) {
            w.ground_on = 0;
            changed = true;
        }
    }
    if (changed) implicit_solve(m, io, w, f0, n, use_contact);

    // post-step: forces, bristles, ledger
    const T* thd1 = w.nu1 + NB;
    T diss_contact = 0;
    if (use_contact) {
        body_twists(m, w.f, w.nu1, w.V);
        T work_in = 0, dVn = 0, dVt = 0;
        for (int c = 0; c < nc; ++c) {
            T vc[3];
            point_velocity(w.V[m.contact_body[c]], w.x[c], vc);
            const T xo0 = io.xi[2 * c + 0], xo1 = io.xi[2 * c + 1];
            const T fz = w.normal[c] ? w.k_n * w.delta[c] - cn_eff * vc[2] : T(0);
            const T fx = (w.stick[c] ? -w.k_t * xo0 - ct_eff * vc[0] : T(0)) + w.fslip[c][0];
            const T fy = (w.stick[c] ? -w.k_t * xo1 - ct_eff * vc[1] : T(0)) + w.fslip[c][1];
            const bool slip = w.normal[c] && !w.stick[c];
            T xn0 = w.stick[c] ? xo0 + h * vc[0] : T(0);
            T xn1 = w.stick[c] ? xo1 + h * vc[1] : T(0);
            if (slip) {
                xn0 = -w.fslip[c][0] / w.k_t;
                xn1 = -w.fslip[c][1] / w.k_t;
            }
            work_in += -h * (fx * vc[0] + fy * vc[1] + fz * vc[2]);
            const T d_old = p_max(w.delta[c], T(0));
            const T d_new = p_max(w.delta[c] - h * vc[2], T(0));
            dVn += T(0.5) * w.k_n * (d_new * d_new - d_old * d_old);
            dVt += T(0.5) * w.k_t * ((xn0 * xn0 + xn1 * xn1) - (xo0 * xo0 + xo1 * xo1));
            io.xi[2 * c + 0] = xn0;
            io.xi[2 * c + 1] = xn1;
            if (io.out_fc) {
                io.out_fc[3 * c + 0] = fx;
                io.out_fc[3 * c + 1] = fy;
                io.out_fc[3 * c + 2] = fz;
            }
            if (io.out_active) io.out_active[c] = w.normal[c];
            if (io.out_slip) io.out_slip[c] = slip;
        }
        diss_contact = work_in - dVn - dVt;
    } else {
        for (int c = 0; c < nc; ++c) {
            if (io.out_fc) io.out_fc[3 * c + 0] = io.out_fc[3 * c + 1] = io.out_fc[3 * c + 2] = 0;
            if (io.out_active) io.out_active[c] = 0;
            if (io.out_slip) io.out_slip[c] = 0;
        }
    }

    if (w.nself > 0) {
        body_twists(m, w.f, w.nu1, w.V);
        T work_in = 0, dV = 0;
        for (int i = 0; i < w.nself; ++i) {
            const int p = w.self_pair[i];
            T delta = 0, nv[3], x[3];
            self_pair(m, w, p, delta, nv, x);
            const T vn = self_normal_speed(m, w, p, nv, x);
            if (w.self_on[i]) work_in += -h * (m.self_k * delta - cs_eff * vn) * vn;
            const T d_new = p_max(delta - h * vn, T(0));
            dV += T(0.5) * m.self_k * (d_new * d_new - delta * delta);
        }
        diss_contact += work_in - dV;
    }

    if (io.ball) {
        T* xb = io.ball;
        T* ub = io.ball + 3;
        T* ob = io.ball + 6;
        const T* u1 = w.ball_u1;  // from the solve; after the boxes' friction it points to uf
        // losses = work of the robot on the ball's elements minus what the ball's tanks gained in this step
        T work_in = 0, dV = 0;
        for (int k = 0; k < w.nball; ++k) {
            T rv = 0;
            for (int d = 0; d < NV; ++d) rv += w.ball_row[k][d] * w.nu1[d];
            const T g = dot3(w.ball_n[k], u1) - rv;
            const T f = w.ball_on[k] ? m.ball_k * w.ball_delta[k] - cb_eff * g : T(0);
            work_in += h * f * rv;
            const T d_new = p_max(w.ball_delta[k] - h * g, T(0));
            dV += T(0.5) * m.ball_k * (d_new * d_new - w.ball_delta[k] * w.ball_delta[k]);
        }
        const T fg = w.ground_on ? m.ball_kg * w.ball_dg - cg_eff * u1[2] : T(0);
        const T dg_new = w.ground_cand ? p_max(w.ball_dg - h * u1[2], T(0)) : T(0);
        dV += T(0.5) * m.ball_kg * (dg_new * dg_new - w.ball_dg * w.ball_dg);
        const T dT = T(0.5) * m.ball_m * (dot3(u1, u1) - dot3(ub, ub));
        const T dG = m.gravity ? -h * m.ball_m * dot3(u1, m.g) : T(0);
        diss_contact += work_in - dT - dG - dV;
        const T I = m.ball_I, r = m.ball_r;
        const T m_eff = T(1) / (T(1) / m.ball_m + r * r / I);
        // Coulomb friction of every touched box at its contact point -r n; the reaction goes to the robot
        T uf[3] = {u1[0], u1[1], u1[2]}, of[3] = {ob[0], ob[1], ob[2]};
        for (int d = 0; d < NV; ++d) w.ball_dp[d] = 0;
        for (int k = 0; k < w.nball; ++k) {
            T rv = 0;
            for (int d = 0; d < NV; ++d) rv += w.ball_row[k][d] * w.nu1[d];
            const T g = dot3(w.ball_n[k], u1) - rv;
            const T f = w.ball_on[k] ? m.ball_k * w.ball_delta[k] - cb_eff * g : T(0);
            int idx[NV];
            T col[NV][3];
            const int nz = point_jacobian(m, w.f, w.ball_body[k], w.ball_x[k], idx, col);
            T vc[3] = {0, 0, 0};
            for (int s = 0; s < nz; ++s)
                for (int i = 0; i < 3; ++i) vc[i] += col[s][i] * w.nu1[idx[s]];
            const T* nk = w.ball_n[k];
            const T rc[3] = {-r * nk[0], -r * nk[1], -r * nk[2]};
            T wr[3], rel[3];
            cross3(of, rc, wr);
            for (int i = 0; i < 3; ++i) rel[i] = uf[i] + wr[i] - vc[i];
            const T rn = dot3(rel, nk);
            for (int i = 0; i < 3; ++i) rel[i] -= rn * nk[i];
            const T sn = p_sqrt(dot3(rel, rel));
            const T Jn = p_min(sn * m_eff, m.ball_mu_robot * f * h) / p_max(sn, T(1e-12));
            const T J[3] = {-Jn * rel[0], -Jn * rel[1], -Jn * rel[2]};
            const T k0 = T(0.5) * m.ball_m * dot3(uf, uf) + T(0.5) * I * dot3(of, of);
            T rJ[3];
            cross3(rc, J, rJ);
            for (int i = 0; i < 3; ++i) {
                uf[i] += J[i] / m.ball_m;
                of[i] += rJ[i] / I;
            }
            diss_contact += k0 - (T(0.5) * m.ball_m * dot3(uf, uf) + T(0.5) * I * dot3(of, of)) + dot3(J, vc);
            for (int s = 0; s < nz; ++s) w.ball_dp[idx[s]] -= dot3(col[s], J);
        }
        // rolling resistance, then Coulomb friction at the lowest point: impulses that only remove kinetic energy
        u1 = uf;
        T ox = of[0], oy = of[1];
        const T ke0 = T(0.5) * m.ball_m * dot3(u1, u1) + T(0.5) * I * dot3(of, of);
        const T on = p_sqrt(ox * ox + oy * oy);
        const T cut = p_min(on, h * m.ball_roll * fg * r / I) / p_max(on, T(1e-12));
        ox -= cut * ox;
        oy -= cut * oy;
        const T s0 = u1[0] - r * oy, s1 = u1[1] + r * ox;
        const T sn = p_sqrt(s0 * s0 + s1 * s1);
        const T Jn = p_min(sn * m_eff, m.ball_mu * fg * h) / p_max(sn, T(1e-12));
        const T J0 = -Jn * s0, J1 = -Jn * s1;
        const T u2[3] = {u1[0] + J0 / m.ball_m, u1[1] + J1 / m.ball_m, u1[2]};
        const T o2[3] = {ox + r * J1 / I, oy - r * J0 / I, of[2]};
        diss_contact += ke0 - (T(0.5) * m.ball_m * dot3(u2, u2) + T(0.5) * I * dot3(o2, o2));
        for (int i = 0; i < 3; ++i) {
            xb[i] += h * w.ball_u1[i];
            ub[i] = u2[i];
            ob[i] = o2[i];
        }
    }

    T supplied = 0, copper = 0, gearbox = 0, stops = 0, mech = 0;
    for (int k = 0; k < NJ; ++k) {
        supplied += w.v[k] * w.i1[k];
        copper += m.mR[k] * w.i1[k] * w.i1[k];
        gearbox += w.damping[k] * thd1[k] * thd1[k];
        if (m.limits_enabled && w.stop_on[k]) stops += (m.lim_c + h * m.lim_k) * thd1[k] * thd1[k];
        if (io.jtau) mech += io.jtau[k] * thd1[k];
    }
    if (use_fext)
        for (int i = 0; i < 3; ++i) mech += f_body[i] * w.nu1[3 + i];
    if (io.fgen)
        for (int d = 0; d < NV; ++d) mech += io.fgen[d] * w.nu1[d];
    io.ledger[0] += h * supplied;
    io.ledger[1] += h * mech;
    io.ledger[2] += h * copper;
    io.ledger[3] += h * gearbox;
    io.ledger[4] += h * stops;
    io.ledger[5] += diss_contact;

    // the robot's IMU: gyro = omega_B(n+1); accelerometer = R^T (a - g) of the trunk origin with
    // a = R_n ((v_B(n+1) - v_B(n)) / h + omega_B(n+1) x v_B(n+1))   (firmware.py imu_step)
    if (io.out_gyro || io.out_acc) {
        T wv[3];
        cross3(w.nu1, w.nu1 + 3, wv);
        T acc[3];
        for (int i = 0; i < 3; ++i) acc[i] = (w.nu1[3 + i] - w.nu[3 + i]) / h + wv[i];
        if (m.gravity) {
            const T* R0 = w.f.R[0];
            for (int i = 0; i < 3; ++i) acc[i] -= R0[0 * 3 + i] * m.g[0] + R0[1 * 3 + i] * m.g[1] + R0[2 * 3 + i] * m.g[2];
        }
        for (int i = 0; i < 3; ++i) {
            if (io.out_gyro) io.out_gyro[i] = w.nu1[i];
            if (io.out_acc) io.out_acc[i] = acc[i];
        }
    }

    // configuration update; p_{n+1} = M(q_n) nu_{n+1}
    if (!m.fixed_base) {
        T dr[3];
        mv3(w.f.R[0], w.nu1 + 3, dr);
        for (int i = 0; i < 3; ++i) q[i] += h * dr[i];
        T rv[3] = {h * w.nu1[0], h * w.nu1[1], h * w.nu1[2]};
        T dq[4], qn[4];
        quat_exp(rv, dq);
        quat_mul(q + 3, dq, qn);
        const T nrm = p_sqrt(qn[0] * qn[0] + qn[1] * qn[1] + qn[2] * qn[2] + qn[3] * qn[3]);
        for (int i = 0; i < 4; ++i) q[3 + i] = qn[i] / nrm;
    }
    for (int k = 0; k < NJ; ++k) th[k] += h * thd1[k];
    for (int a = 0; a < NV; ++a) {
        T s = 0;
        for (int b = 0; b < NV; ++b) s += w.M[a * NV + b] * w.nu1[b];
        io.p[a] = s;
    }
    if (io.ball)
        for (int a = 0; a < NV; ++a) io.p[a] += w.ball_dp[a];
    for (int k = 0; k < NJ; ++k) io.lam[k] = m.mL[k] * w.i1[k];

    if (io.out_nu)
        for (int d = 0; d < NV; ++d) io.out_nu[d] = w.nu1[d];
    for (int k = 0; k < NJ; ++k) {
        if (io.out_i) io.out_i[k] = w.i1[k];
        if (io.out_tau) io.out_tau[k] = w.Ks[k] * w.i1[k];
        if (io.out_v) io.out_v[k] = w.v[k];
    }
}

}  // namespace ph
