// Every constant of the plant in one plain block: copied to the GPU once, read by every robot.
// Filled from Python (ph_k1/native.py) out of K1Model + MotorBank + ContactParams + LimitParams:
// no MJCF parsing in C++.
#pragma once

#include "config.hpp"

namespace ph {

template <class T>
struct ModelData {
    // switches (K1PHPlant constructor arguments and the enabled flags)
    int nbody;
    int ncontact;
    int fixed_base;        // trunk welded: only the 22 hinges are free
    int motors_connected;  // gyrator on/off (off: joint_torque test port)
    int gravity;
    int contact_enabled;
    int limits_enabled;
    int self_enabled;      // self-contact elements on
    int nsphere;
    int npair;

    T h;     // plant time step [s]
    T g[3];  // gravity vector [m/s^2]

    // kinematic tree (MuJoCo order: a parent comes before its children; body 0 is the free trunk)
    int parent[MAX_BODIES];      // -1 for the trunk
    int body_joint[MAX_BODIES];  // hinge 0..21 moving the body, -1 welded, -2 the free trunk
    int jnt_body[NJ];            // body of each hinge
    int jnt_parent_body[NJ];     // parent of that body
    T body_pos[MAX_BODIES][3];   // offset in the parent frame
    T body_rot[MAX_BODIES][9];   // body_quat as a row-major matrix
    T jnt_axis[NJ][3];           // hinge axis, local
    T jnt_pos[NJ][3];            // hinge anchor, local

    // inertia
    T mass[MAX_BODIES];
    T ipos[MAX_BODIES][3];     // COM in the body frame
    T I_body[MAX_BODIES][9];   // rotational inertia about the COM, body frame
    T armature[NJ];

    // joint ranges: the MJCF range is the firmware's command clamp; the mechanical end stops are wider
    // (ph_k1/model.py MEASURED_RANGE: the real K1 went past the MJCF range in its get-ups)
    T jnt_low[NJ];
    T jnt_high[NJ];
    T stop_low[NJ];
    T stop_high[NJ];
    T lim_k, lim_c;  // end-stop spring and damper

    // contact points (fixed to bodies) and their pH elements
    int contact_body[MAX_CONTACTS];
    T contact_local[MAX_CONTACTS][3];
    T k_n, c_n, k_t, c_t, mu;

    // self-contact (ph_k1/self_contact.py): spheres fixed to bodies, unilateral spring + damper per sphere pair
    int sphere_body[MAX_SPHERES];
    T sphere_pos[MAX_SPHERES][3];
    T sphere_r[MAX_SPHERES];
    int pair_a[MAX_SELF_PAIRS];
    int pair_b[MAX_SELF_PAIRS];
    T self_k, self_c;

    // 22 DC motors: L di/dt = v - R i - s k qdot, tau = s k i
    T mk[NJ], mR[NJ], mL[NJ], tau_max[NJ], damping[NJ];
    T v_max;

    // the robot's command path (K1_Deploy firmware safety.cpp apply_safety, once per policy step;
    // ph_k1/firmware.py is the reference): switches and tables
    int fw_pos_clamp;
    int fw_slew;
    int fw_target_clamp;
    int fw_overspeed;
    T policy_dt;                  // k1_config.h kDt
    T fw_effort[NJ];              // effort_limit (= MJCF forcerange)
    T fw_max_joint_velocity[NJ];  // max_joint_velocity (overspeed guard)
    T fw_max_target_velocity[NJ]; // max_target_velocity (slew)
};

}  // namespace ph
