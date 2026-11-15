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

## Files

- `contract.py`: the sprint agent's observation, decode, twist limits and ramp
- `backends.py`: `PHBackend`, `MuJoCoBackend`, and `Walker`, the firmware's loop
- `drive.py`: the slider viewer
- `compare.py`: the scenarios, metrics and report
- `tests/test_walk.py`: `python -m pytest rl/walk/tests -q`; `PH_TEST_DISPLAY=1` also opens the slider panel
