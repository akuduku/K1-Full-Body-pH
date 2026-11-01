// The robot's command path for ONE robot (ph_k1/firmware.py is the reference):
//   once per policy step   K1_Deploy firmware/src/safety.cpp apply_safety: position clamp, slew, target clamp
//                          (driver PD torque within +-effort), overspeed guard -> the target that is sent
//   every plant step       the target enters a delay line; the motor driver uses the one sent `lag` steps ago
#pragma once

#include "model_data.hpp"

namespace ph {

// apply_safety, term by term.  q, qd: the measured joint state at the policy step; last: the previous sent
// target (state, updated); out: the target sent for this policy step.
template <class T>
PH_HD void safety_filter(const ModelData<T>& m, const T* target, T* last, const T* q, const T* qd, const T* kp,
                         const T* kd, T* out) {
    for (int r = 0; r < NJ; ++r) {
        T t = target[r];
        if (m.fw_pos_clamp) t = p_clamp(t, m.jnt_low[r], m.jnt_high[r]);
        if (m.fw_slew) {
            const T max_delta = m.fw_max_target_velocity[r] * m.policy_dt;
            t = p_clamp(t, last[r] - max_delta, last[r] + max_delta);
        }
        if (m.fw_target_clamp && kp[r] > T(0)) {  // the firmware divides by kp (never 0 on the robot)
            const T damping_term = kd[r] * qd[r];
            const T lo = q[r] + (-m.fw_effort[r] + damping_term) / kp[r];
            const T hi = q[r] + (m.fw_effort[r] + damping_term) / kp[r];
            t = p_clamp(t, lo, hi);
        }
        if (m.fw_overspeed) {
            if (qd[r] > m.fw_max_joint_velocity[r])
                t = p_min(t, q[r]);
            else if (qd[r] < -m.fw_max_joint_velocity[r])
                t = p_max(t, q[r]);
        }
        out[r] = t;
        last[r] = t;
    }
}

}  // namespace ph
