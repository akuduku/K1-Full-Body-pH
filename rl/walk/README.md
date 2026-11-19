# Walk: one trained policy on two physics engines

Runs a walking policy trained elsewhere on the pH plant and on MuJoCo, with the same observation, firmware path, delay and
servo table on both. Nothing here trains, so it runs on the laptop.

The policies come from `../k1_sprint/checkpoints/` (`walk_v9`, `soccer-rolling-walk`, `sprint-repro-001`, `lean-B`).

## Drive it

    python rl/walk/drive.py                     # pH plant, walk_v9
    python rl/walk/drive.py --physics mujoco    # same policy, MuJoCo
    python rl/walk/drive.py --physics both      # both at once: pH grey, MuJoCo orange

Three sliders set forward speed, sideways speed and turn rate. Each stays where you leave it. **Stop** sets all three to 0,
**Reset robot** puts it back at the start. The numbers under the sliders are the request, the ramped command and the
measured speed. The limits are the firmware's, and the command follows its ramp (`contract.py`).

In the MuJoCo window: Home stops, R or End resets, Q or Esc quits.

## Compare the physics

    python rl/walk/compare.py [--policy walk_v9] [--quick]

Writes `report.md`, plots and `results.json` to `outputs/compare/<policy>/`.

This shows how the two physics differ, not which is right. For that, `Model Based RL` replays the real get-up logs on both:

    python -m mbrl.replay_mujoco

## The walk task (for training and scoring)

`task.py` is the one place where the task is written: Eric Plass's walk rewards and velocity commands (RL_Core
`k1_skill.py`, `k1_skill_cfg.py`, `k1_sprint/env.py`, the "base" variant) on the pH plant's quantities, plus the
electrical power drawn. Its docstring lists what differs from his environment (no self-collision term, the plant's own
masses and ground). `env.py` runs it on the C++ plant as an rsl-rl environment; the same task on the differentiable
Python plant is `mbrl/grad/walk_env.py` in `Model Based RL`.

    python rl/walk/evaluate.py --policy walk_v9 -B 64 [--quiet]      # pH plant: falls, speed error, power per command
    python rl/walk/eval_mujoco.py --policy walk_v9 [--push 0.3]      # the same policy file in MuJoCo, one robot
    python rl/walk/export.py logs/<run>/model_100.pt                 # checkpoint -> the sprint agent's .bin (75 -> 22)

`evaluate.py` `WalkEval` is the evaluation inside a training loop: the six fixed commands under the training's
conditions, the same at every evaluation (`Model Based RL` `mbrl/grad/train.py --task walk` uses it).  The robot's
sprint agent (`K1_Deploy` `firmware/src/agents/sprint.cpp`) loads the `.bin` that `export.py` writes.

Eric's `walk_v9` in this environment (16 robots, 10 s per command, quiet), a check of the environment: no falls on
stand, 0.5 and 1.0 m/s forward, 0.5 m/s back, 0.5 m/s sideways and a turn of 1 rad/s; speed error 0.05 to 0.10 m/s;
105 W at 0.5 m/s and 182 W at 1.0 m/s.

## Files

- `contract.py`: the sprint agent's observation, decode, twist limits and ramp
- `backends.py`: `PHBackend`, `MuJoCoBackend`, and `Walker`, the firmware's loop
- `drive.py`: the slider viewer
- `compare.py`: the scenarios, metrics and report
- `task.py`: the walk task (rewards, commands, what the rewards read from the plant)
- `env.py`: `WalkEnv`, the task on the C++ plant; `evaluate.py`, `eval_mujoco.py`: scoring
- `export.py`: checkpoint -> the firmware sprint agent's policy file
- `video.py`: a video of a walk policy for W&B (one quiet episode, the requests changing, the request as an arrow)
  with `hamiltonian.py`'s figure of the same episode: the tanks of H and the closed-loop H_d, the energy ledger,
  H_d over (angle, momentum) of the left knee and hip
- `tests/test_walk_env.py`: the task's signals against MuJoCo, single reward terms against Eric's formulas, the
  environment with `walk_v9`, the export round trip, `WalkEval`
- `tests/test_walk.py`: `python -m pytest rl/walk/tests -q`; `PH_TEST_DISPLAY=1` also opens the slider panel
