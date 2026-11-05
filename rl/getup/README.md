# Get up and stay up, with a port-Hamiltonian controller

The second skill on the pH plant: the K1 lies on its back or front, gets up, stays standing under pushes, and
gets up again when it falls. What is new compared to `rl/stand`: the port-Hamiltonian structure is used on the
**agent** side too, not only in the simulator.

| | `rl/stand` | `rl/getup` |
|---|---|---|
| simulator | pH plant | pH plant (+ the gains go through the delay line, + an assist force port) |
| policy output | 22 targets | 22 targets **+ 22 stiffness + 22 damping** (variable impedance) |
| controller | fixed PD | PD as a pH system (virtual spring + damper), **passive by an energy tank** |
| energy in the reward | coil heat R·i² (< 1 % of the reward) | **battery energy** (work + heat), contact and end-stop losses, motor torque-speed margin, kinetic energy, centre-of-mass height from the gravity tank |

## The controller (`contract.py`, `impedance.py`)

The motor driver's PD `τ = kp (t − q) − kd q̇` is a virtual spring `S = ½ Σ kp (t − q)²` plus a damper. With fixed
target and stiffness, robot + controller is passive: energy shaping + damping injection, `S + H_robot` can only
decrease. Getting up needs about **68 J** of potential energy (19.7 kg, centre of mass 0.08 → 0.44 m), and the
controller can only put it in by moving the targets or raising the stiffness: `I = S(new) − S(old)` at the same
`q`. That is exactly what breaks passivity.

The **energy tank** makes it exact and bounded:
- every injection `I > 0` is paid from the tank, and relaxing a spring (`I < 0`) pays back;
- the tank is refilled with what the controller's dampers take out of the robot, `Σ kd q̇² dt`;
- if the tank can't pay, the command is scaled back toward the last one (largest affordable α, bisection).

Over any interval the controller injects at most `T(start) + damping dissipation`, so the closed loop is passive
by construction, whatever the network outputs. The tank level is an observation, so the policy can budget.
Default capacity 150 J; `--no-tank` is the ablation.

## The task (`env.py`), following RL_Core's `K1StandUpEnv`

- **Starts:**
  - a library of lying poses that the pH physics settled at start-up (drops on the back and front with noisy
    joints, held 0.4 s, then compliant), each at a random heading;
  - plus standing starts at HOME;
  - the lying share grows from 50 % to 85 % with the success.
- **Episodes:** 10 s. They end only on the timeout (or a NaN), so the robot gets up, stays up, and gets up again.
- **Curriculum,** every stage driven by the success rate (an EMA), so it also works from scratch and survives
  resumes (saved in every checkpoint):
  - the **assist** (100 N up on lying robots that don't stand yet) fades out per start kind as their success
    reaches 50 %;
  - then the **head penalties** ramp in (the neck breaks on head strikes);
  - then **strong pushes** (×2.5) on standing robots knock them over now and then;
  - the **variable gains** start locked: until both lying starts succeed 50 % of the time the driver gets the
    agent's fixed kKp / kKd (as the stock StandupAgent and RL_Core's get-up recipe), then the 44 gain outputs ramp
    in over 500 iterations (`gain_scale` 0 → 1, in the checkpoint; the action-rate penalty on them ramps with it);
  - the **costs of moving** (battery and impacts) grow from ×0 to ×1 as the success goes from 50 % to 90 %: get up
    first, then cheaply.
  - Why (runs getup-ph-v1 and getup-ph-notank-v1, 2026-11-01/02): with the battery term at ×0.25 from the start and
    the gains free, the lying robots never stood up in 2256 iterations (success 0 %). The battery penalty of a
    get-up attempt was larger than the height and upright rewards (episode sums −22 vs +9), so the policy learnt
    to go limp lying down: stiffness ×0.4, battery 350 → 16 W. The standing starts stayed up (100 %). The tank
    never limited anything. Old behaviour: `--energy-scale-lo 0.25 --gain-free-ema -1`.
- **Rewards:**

| term | weight | from |
|---|---|---|
| height | 3 | centre-of-mass height = the gravity tank `V_grav / (m g)` of H, lying 0 → standing 1 |
| upright | 2 | trunk orientation |
| standing | 6 | bonus while high, upright, both feet down, nothing else touching |
| stillness | −2 /J | kinetic energy T of H while standing |
| battery | −0.03 /W × curriculum | electrical power drawn from the battery, `max(Δ supplied / dt, 0)`: work **and** heat |
| impacts | −0.02 /W × curriculum | power dissipated in the contact elements (impacts, sliding) |
| end_stops | −0.1 /W | power dissipated in the end stops |
| voltage_margin | −1 | `|R τ / k + k q̇| / V > 0.8`: the motor far out on its torque-speed line |
| torque_margin, soft_joint_limits, target_beyond_limits | −2, −2, −0.5 | as RL_Core |
| self_proximity | −2 | the plant has no self-collision: the get-up example's 10 limb pairs, closer than 9 cm |
| action_rate, joint_acc | −0.01, −2.5e-7 | smoothness (the rate covers the gains × the gain curriculum) |
| head_contact, head_impact, neck_torque | −1, −2, −2 × curriculum | as RL_Core |

- **Logged:**
  - success and time to stand per start kind;
  - **battery energy per get-up (J)** and its **efficiency** (potential energy gained / battery energy);
  - standing power (W);
  - tank level, injection and how often it limits;
  - curriculum;
  - videos of get-ups from the back and the front, without the assist.

## Train on free Colab, by time

```bash
colab-train validate colab/k1-getup.yaml
colab-train launch --browser colab/k1-getup.yaml     # or without --browser: it prints the steps
```

Each session runs these steps:
1. `colab/setup_gpu.sh`: packages, CUDA build, GPU tests, about 10 min.
2. `rl/getup/train.py --hours 6.75`: training, then a clean stop (final checkpoint, `deploy.bin`, W&B run
   finished). A checkpoint is written every 10 min, so a disconnect loses at most that.
3. In the next session, **Run all** again continues the same W&B run, including its curriculum.

The rough budget on a T4 is about 35 M steps per hour, about 2 500 PPO iterations per full 7 h session. Plan
**about 2 full sessions** (more if Colab ends them early): the RL_Core and Stand Up get-ups needed 200–300 M
steps. W&B: `karl-jahnel-universit-t-leipzig` /
**Akos pH GetUp**.

```bash
python rl/getup/train.py --device cpu --num-envs 64 --hours 0.02 --no-wandb     # local smoke test
python rl/getup/train.py --no-tank ...            # ablation: variable gains without the tank
python rl/getup/train.py --reward battery=0 ...   # ablation: no energy reward
python rl/getup/train.py --assist-n 0 ...         # no assist
```

## Watch it on the laptop (MuJoCo, sim-to-sim)

```bash
python rl/getup/eval_mujoco.py --policy deploy_1200.bin --start supine --viewer
python rl/getup/eval_mujoco.py --policy deploy_1200.bin --start prone --push 0.5
```

The same agent (observations, gains, tank, filter, delay) runs in MuJoCo, which **has** self-collision. It
prints the time to stand, the battery energy to stand up (estimated with the plant's motor constants), the
potential energy gained, the tank use, and the self-contacts.

## On the robot: what the firmware needs (`standup_ph` agent)

Training and the MuJoCo check need nothing from the robot. The robot's motors already take kp/kd in **every**
motor command (`main.cpp` publishes `LowCmd` with q, kp, kd every 5 ms), but `k1_motion_test` copies an agent's
gains once before the control loop (`kp_run = agent->kp()`, main.cpp:897). A new agent, about 100 lines next to
`StandupAgent` in `src/agents/sprint.cpp`, needs to:

1. **Observe 117 values:** `SprintAgent::observe`'s first 72 values (raw gyro, gravity, `q − kHome`, raw `q̇`)
   with the last action extended to 66, plus the tank level / capacity.
2. **Decode 66 outputs:**
   - targets `kHome + 0.25·kEffort/kKp·a[0:22]`;
   - `kp = kKp·2^(1.5·s·tanh a[22:44])`;
   - `kd = kKd·2^(1.5·s·tanh a[44:66])`, with `s` = the policy's `gain_scale` at the end of its training (1 once
     the gain curriculum has finished; 0 = fixed gains).
3. **Run the tank of `impedance.py` once per 20 ms:**
   - refill `Σ kd q̇² dt`;
   - injection `S(new) − S(old)` at the measured q, with the target after `apply_safety`;
   - scale back with α if the tank can't pay.
4. **Send these kp/kd** in the LowCmd and to `apply_safety` every policy step, instead of the fixed `kp_run`.
5. Keep the target clamp on `k1::effort_limit`: don't override `effort()` with sprint's larger `kEffort`.
6. Like `StandupAgent`: no slew, no tilt abort, never `done()`.
7. **Load the 117 → 66 network**, the same EigenMLPHist `.bin` from `export.py`. The loader must accept 66 outputs.

## Limits, honestly

- **Rigid-body dynamics** are the same as MuJoCo's. The value of pH is in the energy accounting, the motors, the
  controller structure and the rewards.
- **Motor constants** (V_NOM, L/R) are placeholders: watts and joules are model units. Comparisons between
  policies are valid; absolute "J per get-up" needs the real k, R and gear ratio.
- **No self-collision** in the plant: there is the proximity penalty, and the MuJoCo check, which has real
  collisions, shows any exploit.
- **Tank accounting** happens at command time (the transport delay shifts when the driver switches), the motor's
  torque limit can cut the PD torque, and the damper refill is the trapezoid over 20 ms. All of it is computable
  on the robot, which is the point.
- **Stiffness and energy:** holding a pose against gravity costs the same torque at any stiffness, so the tank and
  the gains matter in the dynamic phases (getting up, catching pushes). Energy while standing is set by the pose
  and how calm the control is.
