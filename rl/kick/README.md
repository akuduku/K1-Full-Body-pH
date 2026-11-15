# Walk and kick on the pH plant

The robot walks on velocity commands, then gets a kick request: a ball ahead of it, a direction (up to ±34° off its
heading) and a ball speed (1–6 m/s). It has 4 s to kick and stand again. The task and the agent are Eric Plass's
`k1_kick` (`--variant speed --camera --walk`, kick-014 to kick-016, resting balls) and the firmware's `kick_walk`
agent; the physics is the port-Hamiltonian plant with its ball (`ph_k1/ball.py`).

## A MuJoCo-trained kick on the pH plant

Eric's policies were trained in MuJoCo (mjlab) and run on the robot. Here they run unchanged on the pH plant:

    python rl/kick/evaluate.py --policy ../k1_kick/checkpoints/kick_018.pt -B 512

512 robots, one episode each, no sensor noise or pushes, 20 ms delay, the camera as in training (seed 2026):

| | kick-016 on pH | kick-016 in its own MuJoCo env (k1_kick README) | kick-018 on pH |
|---|---|---|---|
| kicked | 100 % | 99.3 % | 99.8 % |
| direction error | 9.6° | 9.4° | 11.8° |
| near / far balls | 9.5° / 9.7° | 8.2° / 10.0° | 12.6° / 11.4° |
| stopped / mid-walk | 8.0° / 13.7° | 8.0° / 12.7° | 11.7° / 12.1° |
| ball speed at full power | 4.57 m/s | 4.59 m/s | 4.92 m/s |
| falls | 0.2 % | 2.1 % | 0.2 % |

With the training's noise, pushes and delays (`--noise`) kick-018 kicks 98.2 % of the balls at 13.1°.

The two physics engines agree on this task to within the scatter of the policy. The pH plant adds what MuJoCo does
not report: a kick draws about 350–400 J from the battery between the request and the end of the launch, the ball
leaves with 1.5–1.7 J.

`--video DIR` writes a video of one walk and kick and the Hamiltonian figure of that rollout.

## The Hamiltonian figure

`rl/kick/hamiltonian.py`, logged to W&B with every video as `Energy/hamiltonian`. Sprangers, Lopes and Babuška
("Reinforcement learning for port-Hamiltonian systems", Fig. 3b) draw the desired Hamiltonian H_d over the pendulum's
phase plane with the trajectory as dots. The K1 has no single phase plane, so the figure has four panels:

- (a) the tanks of H during the request (kinetic, gravity, contact, coils, ball), their sum H, and
  H_d = H + Σ ½ kp (θ − θ*)²: the motor law is a virtual spring to the target (energy shaping) and a virtual damper
  (damping injection), and the policy moves the targets;
- (b) the ledger: electrical energy supplied = heat in coils, gearboxes and contacts + change of H;
- (c, d) the paper's Fig. 3b for the kicking leg's knee and hip pitch: H_d over (θ, p) of that joint with all other
  coordinates as they are at the touch, and the rollout as dots. It is a slice through a high-dimensional H_d.

## Training

    python rl/kick/train.py --num-envs 4096 --hours 6 --run-name kick-ph-v1 --init-actor ../k1_kick/checkpoints/kick_018.pt
    python rl/kick/train.py --device cpu --num-envs 16 --iterations 3 --no-wandb       # smoke test

Plain PPO on this env (`--init-actor`: continue a MuJoCo-trained policy on the pH plant). The model-based route
(plan the kick in the pH model, then track the plan) is in the Model Based RL repository, `mbrl/kick`.

Rewards: k1_kick's task terms (follow the walk command; launch speed, speed error, direction, loft; back to the
stance and the heading) and safety terms, plus the electrical energy from the pH ledger. Its style terms (which
foot, how the leg swings) are left out.

W&B groups per iteration: `Kick/` (kick rate, launch speed, speed and direction error, share within 5° and 10°,
loft, touch time, falls while walking and kicking, far and mid-walk balls), `Energy/` (the Hamiltonian's tanks, the
ledger's power flows, electrical energy per kick, the ball's energy and their ratio), `Motors/` (leg torque and
current peaks, share at the torque limit, overspeed), `Walk/`, `Policy/`, `Episode_Reward/`; every 30 min `Eval/`
(the policy without action noise) and the video with the Hamiltonian figure.

## Files

- `contract.py`: the agent's 165 inputs and 22 outputs (`phase=True`: a 166th input, the seconds since the request)
- `camera.py`: the head camera's ball reports, copied from k1_kick
- `env.py`: the environment
- `evaluate.py`: one episode per robot and the shot report; `KickEval` for the harness
- `hamiltonian.py`: the figure
- `train.py`, `export.py`, `policy.py`: PPO with the harness, the firmware's policy file, policy files as functions
- `tests/test_kick.py`: `python -m pytest rl/kick/tests -q`

## Limits

- Resting balls only (k1_kick's rolling balls of kick-017/018 are not ported), one ball size (7.5 cm, 0.2 kg).
- The ball's contact parameters are set against MuJoCo's ball (`validation/ball_vs_mujoco.py`), not against the real one.
- The 166-input variant needs the firmware agent to append the request clock.
