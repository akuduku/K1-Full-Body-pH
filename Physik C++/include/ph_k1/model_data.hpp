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

    // end stops
    T jnt_low[NJ];
    T jnt_high[NJ];
    T lim_k, lim_c;

    // contact points (fixed to bodies) and their pH elements
    int contact_body[MAX_CONTACTS];
    T contact_local[MAX_CONTACTS][3];
    T k_n, c_n, k_t, c_t, mu;

    // 22 DC motors: L di/dt = v - R i - s k qdot, tau = s k i
    T mk[NJ], mR[NJ], mL[NJ], tau_max[NJ], damping[NJ];
    T v_max;

    // classical servo (ph_k1/drive.py): slew limit of the target [rad/s]
    T slew;
};

}  // namespace ph
