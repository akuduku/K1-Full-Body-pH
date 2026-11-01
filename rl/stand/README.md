# Stand still on the pH plant

The first skill trained on the port-Hamiltonian K1: stand still in the firmware's start pose and stay up under
pushes. The policy is trained on the pH plant (`ph_k1/native.py`, C++/CUDA) and is built to the **firmware's
`stand` agent** (K1_Deploy `src/agents/stand.cpp`), so the exported file runs on the robot as

```bash
k1_motion_test --task stand --policy deploy.bin
```

with no new firmware code. Do not take it onto the robot before it stands in `eval_mujoco.py` and on the pH
plant, under delay and pushes.

## The contract (`contract.py`, from `include/obs.h` and `include/k1_config.h`)

| | |
|---|---|
| observation (72) | gyro × 0.25, projected gravity, (q − default)[real2sim], q̇ × 0.05 [real2sim], last action |
| action (22) | `target[r] = default_qpos[r] + action_scale[r] · a[sim2real[r]]` (no clip) |
| start pose | `default_qpos`: straight legs, shoulder roll ∓1.3 (arms down) |
| gains | firmware `stiffness / damping` (legs 30/21/18/60, ankles 36, arms and head 4) |
| command path | `ph_k1/firmware.py`: clamp, slew, target clamp, overspeed guard, 10–40 ms delay, motor driver |
| abort | \|roll\| or \|pitch\| > 0.6 rad (the firmware stops the policy) |
| network | 512-256-128 ELU MLP + observation normalizer, exported as EigenMLPHist `.bin` |

## Task (`env.py`)

- Episodes of 10 s start standing still in the start pose (as on the robot after Prepare + countdown), joint
  noise 0.02 rad, soles on the floor. No assist: the untrained robot falls after ~2–4 s, and the abort ends
  the episode, so the samples stay near upright.
- Randomised per episode: transport delay 10–40 ms, kp/kd ±10 % per joint. Pushes: every 4–8 s a horizontal
  trunk velocity change up to 0.3 m/s. Observation noise on the actor's inputs; the critic gets clean ones plus
  the trunk velocity, height and foot contacts.
- Rewards: RL_Core's `K1StandEnv` terms (orientation, height, trunk velocities, joint speed/acceleration, action
  rate, close to the default pose, joint limits, non-foot contacts, both feet down, alive), plus `action_excess`,
  a fall penalty, and the pH plant's **copper loss** R·i² (W of coil heat), logged with the supplied power.

## Train, watch, export

```bash
TASK=stand kaggle/run.sh ITERATIONS=3000 RUN_NAME=stand-ph-v1     # Kaggle T4 (GPU tests first, then training)
TASK=stand kaggle/fetch.sh stand-ph-v1                            # download runs/, logs/

python rl/stand/train.py --device cpu --num-envs 64 --iterations 20 --no-wandb      # local smoke test
TASK=stand kaggle/run.sh ITERATIONS=2000 RUN_NAME=stand-ph-v1b RESUME=stand-ph-v1   # continue a run
python rl/stand/eval_mujoco.py --policy outputs/kaggle/stand-ph-v1/runs/stand-ph-v1/deploy_3000.bin --viewer
python rl/stand/export.py logs/<run>/model_3000.pt deploy.bin
```

Kaggle's 12 h session cap: the kernel gives the training the time left minus 20 min, and the training stops
cleanly before the cap (final checkpoint + `deploy.bin` + artifact, W&B run finished normally). The run folder is
written directly into the kernel output, so even a hard stop keeps checkpoints and videos. `RESUME=<run>`
continues from the newest checkpoint of that W&B run in a new session (iteration numbers continue).

W&B: `karl-jahnel-universit-t-leipzig` / **Akos pH Stand**, one run per training: losses, reward terms,
terminations, energy, a video every 500 iterations, every checkpoint with its `deploy_<it>.bin`, and at the end
the artifact `k1-stand-ph` (`deploy.bin` + checkpoint).

## Tests

```bash
python -m pytest rl/stand/tests -q        # contract == obs.h, export == firmware forward pass, env mechanics
```

Baseline (same command path, 20 ms delay, nothing perturbed, action 0): the robot falls after 3.8 s on the pH
plant and 2.7 s in MuJoCo.
