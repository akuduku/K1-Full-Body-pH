# K1 Full Body — port-Hamiltonian physics for the Booster K1

The whole **Booster K1** humanoid simulated as **one dissipative port-Hamiltonian
system**: free trunk, 22 hinges, 22 DC motors and foot–ground contact. The only
input is the **voltage on the 22 motor coils**. `mj_step` is never called. MuJoCo is
used for only three things:

1. reading the robot description (MJCF) once: masses, inertias, axes, armature, joint ranges, collision boxes;
2. drawing the robot (`mj_kinematics` in the viewer);
3. serving as the reference our numbers are checked against (`checks.py`).

Part of the master thesis *Physics-informed Reinforcement Learning for Soccer
Robots* (Akos Eross, HTWK Leipzig). This repository is the **physics phase**: the
plant is built and validated with voltages chosen by hand (scripts or a classical
servo). RL training on top of it is the next phase.

![servo hold with a sideways push](docs/servo_hold_push.png)

*Servo hold on pH physics, knocked sideways at 20 cm/s at t = 2 s. The kinetic
energy spike is absorbed by coils and contact dampers. The energy ledger
(supplied vs dissipated) stays balanced.*

---

## The model

```
x = (q, p, xi, lambda)
    q      (29)  trunk position r, trunk quaternion Q, 22 hinge angles theta
    p      (28)  p = M(q) nu,   nu = (omega_B, v_B, theta_dot)   (trunk twist in trunk frame)
    xi  (128,2)  tangential "bristles" of the 128 contact points
    lambda (22)  coil fluxes, lambda = L i

H(x) = 1/2 p^T M(q)^-1 p  +  V_grav(q)  +  V_contact(q, xi)  +  V_stops(theta)  +  sum_j lambda_j^2 / (2 L_j)

x_dot = (J(x) - D(x)) dH/dx + G v,        y = G^T dH/dx = i
```

```
        q         p          xi       lambda
q  [ 0        S_q        0        0      ]        D = diag(0, D_p, 0, R)
p  [ -S_q^T   J_gyro(p)  -J_t^T   B K_s  ]        G = [0; 0; 0; I_22]
xi [ 0        J_t        0        0      ]
λ  [ 0        -K_s B^T   0        0      ]
```

| Block | Meaning |
|---|---|
| `S_q` | Kinematics: `r_dot = R v_B`, `Q_dot = ½ Q ⊗ (0, ω_B)`, `theta_dot = theta_dot` |
| `−S_qᵀ` | Forces from positions push on the momenta: gravity, ground springs, end stops, ∂T/∂θ |
| `J_gyro(p)` | Gyroscopic (Lie-algebra) block of the free trunk. Skew on its own, so it only turns the momentum, like a spinning top |
| `J_t` / `−J_tᵀ` | A sliding sole stretches its friction spring, and the spring pulls the foot back |
| `B K_s` / `−K_s Bᵀ` | Motor gyrator: torque `k i` into the joint, back-EMF `k θ̇` into the coil |
| `D_p` | Gearbox damping, contact dampers, end-stop dampers (`⪰ 0`) |
| `R` | Coil resistance (copper loss) |

`J = −Jᵀ` means J only moves energy between the tanks. `D ⪰ 0` only removes it.
`G` is the electrical door. Together:

```
dH/dt = ∇Hᵀ J ∇H − ∇Hᵀ D ∇H + ∇Hᵀ G v  =  vᵀi − (losses)  ≤  vᵀi        (passivity)
```

The kinetic term `∂T/∂θ` is computed **analytically** with spatial algebra:
`∂T/∂θ_k = −H_k · (S_k ×_m V_parent(k))`, where `H_k` is the momentum of the subtree
after hinge k.

### Contact as pH elements

Every contact point has three parts:
- a normal spring `½ k_n δ²`, which is a tank in H;
- a normal damper;
- a tangential "bristle" spring `½ k_t |ξ|²`, coupled to p through a skew block.

The Coulomb cone `|f_t| ≤ μ f_n` is kept by shortening the bristle (elasto-plastic
slip), which can only remove energy. The ground only pushes, never pulls.

Each foot has 4 sole points, the corners of the MJCF collision box. Every other body
with collision geometry gets the 8 corners of its bounding box, so a fallen robot
lies on the floor instead of sinking through it.

### Motors

Each hinge has the equivalent DC motor whose full-voltage torque–speed line passes
through the Booster datasheet corners. The motor models are E6408, E4315, E4310,
E6416, R14 and HT4438:

```
L di/dt = v − R i − s k θ̇,        τ = s k i
k = V_NOM / ω_max,   R = k V_NOM / τ_stall,   L = (L/R) · R
```

`s ≤ 1` caps the torque at the MJCF `forcerange`, which is the deploy limit.

### Integrator

The integrator is semi-implicit and first order:
- Configuration-dependent forces are taken at the start of the step.
- Dampers, the stiffness of contact and end-stop springs, the motor gyrator and the
  coil resistance are implicit. They are solved together as one linear system in
  `(ν, i)`.
- The trunk orientation is updated with the exponential map.

Implicit Euler on the lossless gyrator coupling can only *remove* energy, never
create it.

---

## Install and run

```bash
git clone https://github.com/akuduku/K1-Full-Body-pH.git
cd K1-Full-Body-pH
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python checks.py                              # validation, about 1 min
python play_volts.py joints --view            # each joint group gets a voltage ramp (trunk welded)
python play_volts.py drop                     # dropped from 5 cm with 0 V (shorted coils)
python hold_pose.py --gain 3 --seconds 10     # classical servo holds the stand pose
python hold_pose.py --gain 3 --push 20        # ... and is knocked sideways at 20 cm/s
python hold_pose.py --gain 1 --view           # firmware gains: tips over (see "Limits")
```

Plots (energy tanks, ledger, trunk height and tilt) and CSVs are written to
`outputs/`. `--view` opens the MuJoCo viewer and needs a display.

```python
from ph_k1.plant import K1PHPlant
plant = K1PHPlant(batch=64)          # 64 robots in parallel
plant.reset()                         # stand pose, soles on the floor
info = plant.step(volts)              # volts: (64, 22) tensor
plant.energy()                        # tanks of H
plant.ledger                          # supplied / dissipated energy
plant.ph_matrices()                   # explicit J, D, G, ∇H for robot 0
```

---

## Repository layout

| Path | Content |
|---|---|
| `ph_k1/model.py` | Reads the MJCF once into torch tensors (tree, masses, contact points) |
| `ph_k1/spatial.py` | Spatial vectors and quaternions |
| `ph_k1/dynamics.py` | Forward kinematics, `M(q)`, analytic `∂T/∂θ`, gyroscopic block, gravity |
| `ph_k1/motors.py` | 22 DC motors from the datasheet torque–speed curves |
| `ph_k1/contact.py` | Contact and end-stop parameters |
| `ph_k1/plant.py` | **`K1PHPlant`**: state, `step(volts)`, `energy()`, energy ledger, `ph_matrices()` |
| `ph_k1/drive.py` | Classical servo: target angles → volts (not RL) |
| `ph_k1/render.py` | Viewer (drawing only) and energy-tank plots |
| `checks.py` | Validation suite |
| `play_volts.py`, `hold_pose.py` | Hand-commanded experiments |
| `assets/K1/` | Booster K1 MJCF and meshes |
| `docs/` | Figures and notes |

The plant is batched: N robots step in parallel in PyTorch. Measured on CPU: about
3–4 ms per step for one robot, about 130–200 µs per robot in batches of 64–256.

---

## Validation (`checks.py`)

| # | Check | Result |
|---|---|---|
| 1 | `M(q)` against MuJoCo `mj_fullM` (random poses) | 2e-14 |
| 2 | Analytic `∂T/∂θ` against autograd | 2e-15 relative |
| 3 | `J = −Jᵀ` exactly, `D ⪰ 0`, `(J−D)∇H + Gv` (∇H by autograd) equals what `step()` integrates, `y = i` | 3e-15 |
| 4 | Accelerations against MuJoCo `mj_forward` under joint torques | 3e-10 relative |
| 5 | Zero-gravity flight without losses: momentum and energy errors shrink linearly with h | first order |
| 6 | Drop onto the floor at 0 V: `ΔH = supplied − dissipated`, residual shrinks with h | about 4% of the losses at h = 0.5 ms |
| 7 | Every motor type follows its datasheet torque–speed envelope | 1e-16 |
| 8 | Welded trunk, right arm: same trajectory as the separate 4-DOF arm plant | 0.63° at 2 ms. Runs only in the thesis folder, SKIP here |
| 9 | Servo holds the stand pose for 3 s on voltages only | tilt 0.9°, ΣF_z = mg, residual −0.015 J |

**Two observations about the energy balance:**
- **Standing still:** energy supplied ≈ copper loss (140.2 J vs 140.2 J over 10 s).
  H stays constant, so holding a pose turns every joule into coil heat.
- **Coils at 0 V:** supplied = 0. When the robot falls, its kinetic energy flows
  back through the gyrator and is burned in R. That is why a shorted motor acts as a
  brake.

![drop at 0 V](docs/drop.png)

---

## Limits and placeholders

- **Motor constants:** `V_NOM = 7 V`, `L/R = 50 ms` and gearbox damping 0.08 N·m·s/rad
  are placeholders. Torque against speed matches the datasheet, but volts and amperes
  are internal units. Real `k`, `R`, `L` and the gear ratio still have to come from
  Booster.
- **Contact numbers:** `k_n = 3e4` N/m per point (about 0.8 mm sink), `μ = 0.7`. They
  are not identified on the real robot.
- **No self-collision.** Only body–ground contact and end stops are modelled.
- **First-order integrator.** Standing and slow motion close the energy balance to
  about 0.01 J. Hard impacts at h = 1 ms leave about 10–15% of the impact loss
  unaccounted, 4% at 0.25 ms.
- **The K1 firmware gains (`--gain 1`) do not keep the robot up.** As an inverted
  pendulum it needs about `m g h ≈ 85 N·m/rad` of ankle stiffness, and two ankles at
  50 N·m/rad each, minus the give of knees and hips, is not enough. This is physics,
  not a bug, and it is what an RL policy will have to fix.

![firmware gains: the robot tips over](docs/firmware_gains_fall.png)

---

## Background

The predecessor code moved the robot with MuJoCo (`mj_step`). Its "port-Hamiltonian"
parts (twist and wrench ports, motor port, a `hamiltonian` observation) were only an
energy log on top of that. `docs/miert-nem-ph-fizika.pdf` explains this (in
Hungarian). This repository writes out H, J, D and G for the full robot and
integrates them.

**Compared with the earlier right-arm pH plant**, the full-body plant:
- solves the gyrator coupling implicitly instead of with forward Euler;
- models end stops as pH springs instead of clipping the joint (which pushed the other joints);
- replaces the MuJoCo contact oracle with pH contact elements;
- computes `∂M/∂q` analytically instead of with finite differences.

**Next phase (RL):** wrap `K1PHPlant` in an environment with these properties:
- **Action:** 22 voltages at 50 Hz.
- **Observation:** gravity direction, ω, θ, θ̇, i.
- **Reward:** upright and height, with an electrical-power penalty `vᵀi`.

`ph_k1/drive.py` can serve as the teacher or baseline.

The Booster K1 model files in `assets/K1/` come from Booster Robotics.
