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
};

template <class T> PH_HD T gyrator_scale(const ModelData<T>& m, int k, T i) {
    const T ki = p_abs(m.mk[k] * i);
    return ki > m.tau_max[k] ? m.tau_max[k] / p_max(ki, T(1e-30)) : T(1);
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
        const T cap = p_min(avail, m.tau_max[k]);
        tau = p_max(p_min(tau, cap), -cap);
        const T i_star = tau / m.mk[k];
        const T s = gyrator_scale(m, k, i);
        const T v = m.mL[k] * (i_star - i) / m.h + m.mR[k] * i_star + s * m.mk[k] * qd;
        w.v[k] = p_clamp(v, -m.v_max, m.v_max);
    }
}

// The implicit solve for the given contact modes -> w.nu1, w.i1   (plant.py step.solve)
template <class T>
PH_HD void implicit_solve(const ModelData<T>& m, const StepIO<T>& io, Work<T>& w, int f0, int n, bool use_contact) {
    const T h = m.h;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) w.A[a * NV + b] = w.M[(a + f0) * NV + (b + f0)];
    for (int k = 0; k < NJ; ++k) w.A[(NB + k - f0) * NV + (NB + k - f0)] += h * w.diag[k];
    for (int d = 0; d < NV; ++d) w.Fc[d] = w.F[d];

    if (use_contact) {
        const T cn_eff = m.c_n + h * m.k_n, ct_eff = m.c_t + h * m.k_t;
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
            T fz = w.normal[c] ? m.k_n * w.delta[c] : T(0);
            T fx = (w.stick[c] ? -m.k_t * io.xi[2 * c + 0] : T(0)) + w.fslip[c][0];
            T fy = (w.stick[c] ? -m.k_t * io.xi[2 * c + 1] : T(0)) + w.fslip[c][1];
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

    // hinge diagonal: gearbox damping + end stops
    for (int k = 0; k < NJ; ++k) {
        w.diag[k] = m.damping[k];
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

    for (int k = 0; k < NJ; ++k) w.Ks[k] = m.motors_connected ? gyrator_scale(m, k, w.i_n[k]) * m.mk[k] : T(0);

    implicit_solve(m, io, w, f0, n, use_contact);

    const T cn_eff = m.c_n + h * m.k_n, ct_eff = m.c_t + h * m.k_t;
    if (use_contact && any_cand) {  // Coulomb cone and unilateral check, one corrected solve if needed
        body_twists(m, w.f, w.nu1, w.V);
        bool changed = false;
        for (int c = 0; c < nc; ++c) {
            uint8_t n2 = 0, s2 = 0;
            T fs0 = 0, fs1 = 0;
            if (w.cand[c]) {
                T vc[3];
                point_velocity(w.V[m.contact_body[c]], w.x[c], vc);
                const T fz = m.k_n * w.delta[c] - cn_eff * vc[2];
                const T ft0 = -m.k_t * io.xi[2 * c + 0] - ct_eff * vc[0];
                const T ft1 = -m.k_t * io.xi[2 * c + 1] - ct_eff * vc[1];
                n2 = fz > T(0);
                const T fz_c = n2 ? p_max(fz, T(0)) : T(0);
                const T ftn = p_sqrt(ft0 * ft0 + ft1 * ft1);
                s2 = n2 && (ftn <= m.mu * fz_c);
                if (n2 && !s2) {
                    const T sc = m.mu * fz_c / p_max(ftn, T(1e-12));
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
        if (changed) implicit_solve(m, io, w, f0, n, use_contact);
    }

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
            const T fz = w.normal[c] ? m.k_n * w.delta[c] - cn_eff * vc[2] : T(0);
            const T fx = (w.stick[c] ? -m.k_t * xo0 - ct_eff * vc[0] : T(0)) + w.fslip[c][0];
            const T fy = (w.stick[c] ? -m.k_t * xo1 - ct_eff * vc[1] : T(0)) + w.fslip[c][1];
            const bool slip = w.normal[c] && !w.stick[c];
            T xn0 = w.stick[c] ? xo0 + h * vc[0] : T(0);
            T xn1 = w.stick[c] ? xo1 + h * vc[1] : T(0);
            if (slip) {
                xn0 = -w.fslip[c][0] / m.k_t;
                xn1 = -w.fslip[c][1] / m.k_t;
            }
            work_in += -h * (fx * vc[0] + fy * vc[1] + fz * vc[2]);
            const T d_old = p_max(w.delta[c], T(0));
            const T d_new = p_max(w.delta[c] - h * vc[2], T(0));
            dVn += T(0.5) * m.k_n * (d_new * d_new - d_old * d_old);
            dVt += T(0.5) * m.k_t * ((xn0 * xn0 + xn1 * xn1) - (xo0 * xo0 + xo1 * xo1));
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

    T supplied = 0, copper = 0, gearbox = 0, stops = 0, mech = 0;
    for (int k = 0; k < NJ; ++k) {
        supplied += w.v[k] * w.i1[k];
        copper += m.mR[k] * w.i1[k] * w.i1[k];
        gearbox += m.damping[k] * thd1[k] * thd1[k];
        if (m.limits_enabled && w.stop_on[k]) stops += (m.lim_c + h * m.lim_k) * thd1[k] * thd1[k];
        if (io.jtau) mech += io.jtau[k] * thd1[k];
    }
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
