# The K1 pH plant in C++/CUDA

`ph_k1/plant.py` (`K1PHPlant`) written again in C++, so that RL can train on it: the same port-Hamiltonian
model (free trunk, 22 hinges, 128 contact points with bristles and the Coulomb cone, 22 DC motors), the same
integrator, the same numbers. Around it sits the **real K1's command path**, so a policy trained here commands
the plant exactly the way it will command the robot. The Python versions stay the references.

- **One CUDA thread = one robot.** The physics is header-only C++ (`include/ph_k1/`) that compiles with
  g++ (CPU, OpenMP over robots) and with nvcc (GPU) from the same source, in float32 or float64.
- **One call per policy step.** `policy_step(targets)` runs the command path and the 20 plant steps of a 50 Hz
  policy step inside one C++ call / one kernel launch.
- **State in PyTorch tensors**, updated in place (zero copy): `q (B,29)`, `p (B,28)`, `xi (B,128,2)`,
  `lam (B,22)`, the ledger `(B,6)`, and the command path's `last_target (B,22)`, delay line `ring (B,D,22)`,
  `ring_idx`, `lag (B,)`.
- **No MJCF parsing in C++**: `ph_k1/model.py` reads the XML; `ph_k1/native.py` sends the numbers once.
- Cheaper tree algorithms with the same results: the mass matrix by composite rigid bodies, generalised
  forces by one backward pass of body wrenches, contact Jacobians only for points in contact, no host syncs
  (the contact mode switch is decided per robot).

## The command path (what the robot does with a policy's output)

Copied from the robot's software, the K1_Deploy firmware (`k1_motion_test`: `firmware/src/main.cpp`,
`firmware/src/safety.cpp`, tables from `firmware/include/k1_config.h`). Python reference: `ph_k1/firmware.py`.

```
policy: 22 joint targets [rad], 50 Hz
  -> once per policy step (safety.cpp apply_safety)    firmware.hpp
       1 position clamp to the firmware range (= MJCF range)
       2 slew |t - t_last| <= max_target_velocity * 0.02 s       (on/off per agent, like agent->slew())
       3 target clamp: the driver's PD torque kp (t - q) - kd qdot within +-effort
       4 overspeed guard: qdot > max_joint_velocity -> t <= q
  -> held for the policy step (the robot sends the same LowCmd every 5 ms)
  -> delay line: the driver sees the target sent `lag` plant steps ago    launch.hpp run_robot
  -> motor driver every 1 ms: PD, limited by the motor's torque at this speed -> volts    step.hpp
  -> pH plant: coil currents, torques, rigid bodies, contact
```

Volts never leave the simulator: the robot has no voltage input. The switches, gains and delay are
`plant.firmware` (`FirmwareConfig`):

| Field | Default | |
|---|---|---|
| `gains` | `walk` | `GAIN_PRESETS`: `walk` (genesis_walk), `firmware`, `stiff_arms`, `hold`, `goalie`, `prepare` (k1_config.h). Use the gains of the agent that will run the policy |
| `pos_clamp`, `slew`, `target_clamp`, `overspeed_guard` | on | the firmware's four steps |
| `delay_ms` | (10, 40) | transport delay, drawn per robot at reset; `plant.set_delay(lo, hi)` re-draws |
| `delay_hold_prob` | None | None: one delay per episode; p: keep it with probability p each policy step |
| `policy_dt` | 0.02 | the firmware's kDt |

Outputs per step, in `info.extra`: `nu` (trunk twist + joint speeds), `gyro` and `acc` (the robot's IMU:
trunk angular velocity and specific force in the trunk frame, standing ≈ (0, 0, +9.81), lying on the back
≈ (+9.81, 0, 0) as in the robot's log), `target` (what the driver held), `volts`.

**Evidence from the real K1** (`Karls Repo/Logs/30102026`, front get-up of 25 Oct, 500 Hz, replayed through the
plant with the trunk fixed; best-fit command → joint lag over the 12 most-moved joints):

| plant setting | mean lag − real |
|---|---:|
| slew on, no delay | +36 ms |
| slew off, no delay | −18 ms |
| **slew off, 20 ms delay** | **+5 ms** |
| slew off, 40 ms delay | +30 ms |

So that get-up ran without the slew limiter and with about 20 ms transport delay; the arms then match within
a few ms, the hips stay 30–40 ms slower (the legs pushed on the floor in the real get-up; the replay holds the
trunk). The end stops are wider than the MJCF range, because the robot went past it (`ph_k1/model.py`
`MEASURED_RANGE`, `docs/robot_compatibility_for_ai.md`).

## Build

```bash
cd ph_k1_cuda && python setup.py build_ext --inplace       # -> ph_k1_native*.so next to setup.py
```

CUDA is built when nvcc exists (Kaggle, Colab), CPU-only otherwise (this laptop). `PH_K1_CUDA=1` requires
CUDA, `PH_K1_CUDA=0` forces CPU. Architectures: `TORCH_CUDA_ARCH_LIST` (default `7.5;8.0;8.6;8.9`).

## Use

```python
from ph_k1.firmware import FirmwareConfig
from ph_k1.model import load_k1
from ph_k1.native import K1PHPlantNative

plant = K1PHPlantNative(model=load_k1(dtype=torch.float32, device="cuda"), batch=4096,
                        firmware=FirmwareConfig(gains="walk", slew=True, delay_ms=(10.0, 40.0)))
plant.reset()
info = plant.policy_step(targets)        # targets (B,22) joint angles: one 50 Hz policy step
info.extra["gyro"], info.extra["acc"]    # the IMU, as the robot measures it
plant.theta, info.extra["nu"][:, 6:]     # joint angles and speeds
info = plant.step(volts)                 # or the plant alone: one plant step with coil volts (B,22)

# a policy that sets its own gains every step (rl/getup): kp/kd (B,22) per call; with delay_gains they travel
# through the delay line with the targets, as in the robot's LowCmd
plant = K1PHPlantNative(..., firmware=FirmwareConfig(gains="standup", slew=False, delay_gains=True))
info = plant.policy_step(targets, kp=kp, kd=kd)
# an external world force on the trunk origin during the step (a training assist); its work -> ledger.mech_input
info = plant.policy_step(targets, trunk_force=f)      # f (B,3) [N]; also plant.step(volts, trunk_force=f)
```

`K1PHPlantNative` is a subclass of `K1PHPlant`: `reset`, `kinematics`, `velocity`, `energy`, `ledger`,
`ph_matrices` work as before. `python validation/checks.py --impl native` runs the whole validation suite with it.

## Tests and results

```bash
python -m pytest ph_k1_cuda/tests -q                       # 21 tests: C++ vs Python, float64, CPU
PH_TEST_DEVICE=cuda python -m pytest ph_k1_cuda/tests -q   # the same on a GPU
python validation/checks.py --impl native                               # 9/9 PASS
python benchmarks/benchmark_ph.py --impl native              # speed
```

- the plant: one step with stick, slip and lift-off agrees with Python to ~1e-13; 300 steps of a drop, the
  welded trunk past its end stops, flight, saturated motors to ~1e-8
- the command path against `ph_k1/firmware.py`: every switch on and off, a different delay per robot
  (0/7/23/40 ms), targets the filter moves by over 1 rad: ~5e-13; the slew rate, the target clamp and the
  overspeed guard each checked; a robot with delay d does exactly what an undelayed one did d ms earlier
- the end stops contain every joint angle the real robot reached; the IMU reads (0, 0, g) standing,
  (+g, 0, 0) lying on the back, 0 in free fall
- float32 on the default command path (10–40 ms delay) holds the stand pose (gain ×3, tilt 0.9°) with the
  energy-ledger residual 0.015 J
- per-step gains through the delay line == the Python reference; a robot with delay d and gains changing every
  step does exactly what an undelayed one did d ms earlier (and without the gain line it does not)
- the trunk force == the Python reference; 250 N lifts the robot off, its work F·Δz goes into mech_input and
  the ledger balances

Laptop CPU (6 threads), float32, policy steps per second (50 Hz policy = 20 plant steps, with the command path):

| robots | PyTorch plant | C++ plant |
|---:|---:|---:|
| 1 | 9 | 1,428 |
| 256 | 192 | 8,065 |
| 1024 | 218 | 8,895 |

On the GPU: `kaggle/run.sh` (Kaggle) or `colab/ph_bench_colab.ipynb` build the CUDA version, run the tests
on the GPU, benchmark it next to the PyTorch plant and MJX, and upload a video of the servo hold + push to W&B.
