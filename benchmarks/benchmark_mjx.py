"""MJX reference speed for the same K1 model, to compare with ``benchmark_ph.py``.

Times ``mjx.step`` batched over ``B`` robots on the GPU (``jax.vmap`` + ``jit``,
the substeps of one 50 Hz policy step fused with ``lax.fori_loop``). The robots
start from a state settled on the floor by plain MuJoCo, so contacts are active
as in training. No policy, no PD: this is the raw cost of the physics.

    python benchmarks/benchmark_mjx.py
    python benchmarks/benchmark_mjx.py --timesteps 0.005,0.001 --batches 1024,4096 --json mjx.json

The default model has primitive collision shapes (the plain K1_22dof.xml has
cylinder-mesh contacts that MJX cannot do); it is a copy of RL_Core's K1_22dof_mjx_prims.xml.
0.005 s is what RL_Core trains with on MJX/mjlab (4 substeps per policy step);
0.001 s matches the pH plant (20 substeps), for an apples-to-apples number.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_XML = Path(__file__).resolve().parents[1] / "assets" / "K1" / "K1_22dof_mjx_prims.xml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--batches", default="256,1024,4096,8192")
    ap.add_argument("--timesteps", default="0.005,0.001", help="physics steps [s] to time")
    ap.add_argument("--policy-dt", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=5, help="timed policy steps per configuration")
    ap.add_argument("--settle-s", type=float, default=0.5, help="CPU MuJoCo settling before the copy")
    ap.add_argument("--target", type=float, default=1e8)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    dev = jax.devices()[0]
    print(f"jax {jax.__version__}  mujoco {mujoco.__version__}  device={dev.platform} ({dev.device_kind})")
    report = {"jax": jax.__version__, "mujoco": mujoco.__version__, "device": dev.device_kind,
              "xml": args.xml.name, "speed": []}

    print(f"\n{'dt [ms]':>8} {'batch':>6} {'ms/physics step':>16} {'ms/policy step':>15} "
          f"{'policy steps/s':>15} {'time for ' + f'{args.target:.0e}':>14}")
    for h in (float(x) for x in args.timesteps.split(",")):
        m = mujoco.MjModel.from_xml_path(str(args.xml))
        m.opt.timestep = h
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        for _ in range(int(args.settle_s / h)):  # land on the floor: contacts active while timing
            mujoco.mj_step(m, d)
        substeps = int(round(args.policy_dt / h))
        try:
            mm = mjx.put_model(m)
            d0 = mjx.put_data(m, d)
        except Exception as exc:  # unsupported geometry / solver option in this MJX version
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            print(f"{h * 1e3:>8.1f}  put_model failed: {msg}")
            report["speed"].append({"timestep": h, "error": msg})
            continue

        def policy_step(data, mm=mm, substeps=substeps):
            return jax.lax.fori_loop(0, substeps, lambda _, x: mjx.step(mm, x), data)

        step = jax.jit(jax.vmap(policy_step))
        for B in (int(b) for b in args.batches.split(",")):
            try:
                batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B, *x.shape)), d0)
                batch = step(batch)  # compile + warm up
                jax.block_until_ready(batch)
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    batch = step(batch)
                jax.block_until_ready(batch)
                sec = (time.perf_counter() - t0) / args.iters
                finite = bool(jnp.isfinite(batch.qpos).all())
            except Exception as exc:  # OOM at large batches
                msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                print(f"{h * 1e3:>8.1f} {B:>6}  failed: {msg[:80]}")
                report["speed"].append({"timestep": h, "batch": B, "error": msg})
                continue
            sps = B / sec
            hours = args.target / sps / 3600
            eta = f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} d"
            print(f"{h * 1e3:>8.1f} {B:>6} {sec / substeps * 1e3:>16.3f} {sec * 1e3:>15.2f} "
                  f"{sps:>15.0f} {eta:>14}{'' if finite else '  (NaN!)'}", flush=True)
            report["speed"].append({"timestep": h, "batch": B, "substeps": substeps,
                                    "ms_physics_step": sec / substeps * 1e3, "ms_policy_step": sec * 1e3,
                                    "policy_steps_per_s": sps, "hours_for_target": hours,
                                    "target": args.target, "finite": finite})
            del batch

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
