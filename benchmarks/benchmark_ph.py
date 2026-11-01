"""Speed benchmark for the port-Hamiltonian K1 plant (``Full Body/ph_k1``).

Answers one question before the pH backend is wired into ``K1BaseEnv``: how
many *policy* steps per second does ``K1PHPlant`` deliver, and how does that
scale with the number of parallel robots and the float type?

Each policy step (50 Hz) is ``--substeps`` plant steps (1 ms each by default)
of: servo (``K1Drive``: target angles -> volts) + ``K1PHPlant.step(volts)``.
The robots hold the stand pose, so the contact solver sees the realistic
case (soles loaded, occasional mode switches).

    python benchmarks/benchmark_ph.py
    python benchmarks/benchmark_ph.py --batches 1,64,256,1024 --dtypes float64,float32
    python benchmarks/benchmark_ph.py --device cuda --batches 256,1024,4096   # Kaggle
    python benchmarks/benchmark_ph.py --profile --batches 64                 # where the time goes
    python benchmarks/benchmark_ph.py --json results.json                    # machine-readable copy

For float32 the script also checks physics quality against float64: trunk
height after ``--hold-s`` seconds and the energy-ledger residual
``H(t) - H(0) - (supplied - dissipated)`` (≈ 0 for a correct pH integrator).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

DEFAULT_PH_PATH = Path(__file__).resolve().parents[1]  # the repo root, which holds ph_k1/


def _import_ph(ph_path: Path, impl: str = "torch"):
    sys.path.insert(0, str(ph_path))
    from ph_k1.drive import K1Drive
    from ph_k1.model import load_k1
    from ph_k1.plant import K1PHPlant

    if impl == "native":  # the C++/CUDA plant (ph_k1_cuda), built with setup.py build_ext --inplace
        from ph_k1.native import K1PHPlantNative as K1PHPlant
    return K1PHPlant, K1Drive, load_k1, impl


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


class Runner:
    """One policy step on the robot's command path (ph_k1/firmware.py: firmware filter, 10-40 ms delay, motor
    driver) + `substeps` plant steps.  torch: firmware.K1Firmware + K1PHPlant.step in Python; native:
    K1PHPlantNative.policy_step, the whole policy step in one C++/CUDA call."""

    def __init__(self, plant, firmware, native):
        self.plant, self.firmware, self.native = plant, firmware, native

    def policy_step(self, target, substeps):
        (self.plant if self.native else self.firmware).policy_step(target, substeps)


def build(ph, batch: int, dtype: torch.dtype, device: str, timestep: float, gain: float = 1.0):
    K1PHPlant, _K1Drive, load_k1, impl = ph
    from ph_k1.firmware import FirmwareConfig, K1Firmware

    cfg = FirmwareConfig(gain_scale=gain)
    model = load_k1(dtype=dtype, device=device)
    if impl == "native":
        plant = K1PHPlant(model=model, batch=batch, timestep=timestep, firmware=cfg)
        return Runner(plant, None, True)
    plant = K1PHPlant(model=model, batch=batch, timestep=timestep)
    return Runner(plant, K1Firmware(plant, cfg), False)


def time_policy_steps(run, substeps: int, warmup: int, iters: int, device: str):
    target = run.plant.theta.clone()
    for _ in range(warmup):
        run.policy_step(target, substeps)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        run.policy_step(target, substeps)
    _sync(device)
    return (time.perf_counter() - t0) / iters  # seconds per policy step


def hold_quality(run, seconds: float, substeps: int):
    """Hold the stand pose; return trunk z, max tilt and the ledger residual."""
    plant = run.plant
    target = plant.theta.clone()
    H0 = plant.energy()["total"].clone()
    for _ in range(max(1, int(round(seconds / (plant.h * substeps))))):
        run.policy_step(target, substeps)
    H1 = plant.energy()["total"]
    led = plant.ledger
    residual = (H1 - H0) - (led.supplied - led.dissipated)
    quat = plant.q[:, 3:7]
    # tilt of the trunk z axis from vertical: cos = 1 - 2 (qx^2 + qy^2)
    cos_tilt = (1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)).clamp(-1, 1)
    return {
        "trunk_z": float(plant.q[:, 2].mean()),
        "tilt_deg": float(torch.rad2deg(torch.arccos(cos_tilt)).max()),
        "residual_J": float(residual.abs().max()),
        "supplied_J": float(led.supplied.mean()),
        "finite": bool(torch.isfinite(plant.q).all()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ph-path", type=Path, default=DEFAULT_PH_PATH, help="folder that contains ph_k1/")
    ap.add_argument("--impl", choices=("torch", "native"), default="torch",
                    help="torch: ph_k1/plant.py; native: the C++/CUDA plant (ph_k1_cuda)")
    ap.add_argument("--batches", default="1,16,64,256,1024")
    ap.add_argument("--dtypes", default="float64,float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--timestep", type=float, default=1e-3, help="plant step [s]")
    ap.add_argument("--policy-dt", type=float, default=0.02, help="policy step [s] (50 Hz)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3, help="timed policy steps per configuration")
    ap.add_argument("--hold-s", type=float, default=2.0, help="seconds for the quality check (0 = skip)")
    ap.add_argument("--quality-batch", type=int, default=4)
    ap.add_argument("--gain", type=float, default=3.0,
                    help="servo gain scale for the quality check (1 = firmware gains, which tip over)")
    ap.add_argument("--profile", action="store_true", help="torch profiler table for the first config")
    ap.add_argument("--profile-sort", default="",
                    help="profiler sort key (default: self_cpu_time_total, on CUDA self_cuda_time_total)")
    ap.add_argument("--json", type=Path, default=None, help="also write all results to this JSON file")
    ap.add_argument("--target", type=float, default=1e8, help="policy steps for the training-time estimate")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    ph = _import_ph(args.ph_path, args.impl)
    substeps = int(round(args.policy_dt / args.timestep))
    batches = [int(b) for b in args.batches.split(",")]
    dtypes = [getattr(torch, d) for d in args.dtypes.split(",")]

    gpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
    report = {"impl": args.impl, "torch": torch.__version__, "device": args.device, "gpu": gpu, "cpu": platform.processor(),
              "threads": torch.get_num_threads(), "timestep": args.timestep, "substeps": substeps,
              "speed": [], "quality": []}
    print(f"impl={args.impl}  torch {torch.__version__}  device={args.device}{f' ({gpu})' if gpu else ''}  "
          f"threads={torch.get_num_threads()}  plant dt={args.timestep * 1e3:.1f} ms  "
          f"substeps/policy step={substeps}")

    if args.profile:
        run = build(ph, batches[0], dtypes[0], args.device, args.timestep)
        target = run.plant.theta.clone()
        acts = [torch.profiler.ProfilerActivity.CPU]
        if args.device.startswith("cuda"):
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=acts) as prof:
            run.policy_step(target, substeps)
        cuda = args.device.startswith("cuda")
        sort = args.profile_sort or ("self_cuda_time_total" if cuda else "self_cpu_time_total")
        print(f"profile: one policy step ({substeps} plant steps), batch {batches[0]}, {dtypes[0]}")
        print(prof.key_averages().table(sort_by=sort, row_limit=25))
        del run

    print(f"\n{'dtype':>8} {'batch':>6} {'ms/plant step':>14} {'ms/policy step':>15} "
          f"{'policy steps/s':>15} {'time for ' + f'{args.target:.0e}':>14}")
    for dtype in dtypes:
        for B in batches:
            try:
                run = build(ph, B, dtype, args.device, args.timestep)
                sec = time_policy_steps(run, substeps, args.warmup, args.iters, args.device)
            except RuntimeError as exc:  # OOM, singular matrix in float32, ...
                msg = str(exc).splitlines()[0][:200]
                print(f"{str(dtype)[6:]:>8} {B:>6}  failed: {msg[:80]}")
                report["speed"].append({"dtype": str(dtype)[6:], "batch": B, "error": msg})
                run = None
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue
            sps = B / sec
            hours = args.target / sps / 3600
            eta = f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} d"
            print(f"{str(dtype)[6:]:>8} {B:>6} {sec / substeps * 1e3:>14.2f} {sec * 1e3:>15.1f} "
                  f"{sps:>15.1f} {eta:>14}", flush=True)
            report["speed"].append({"dtype": str(dtype)[6:], "batch": B, "ms_plant_step": sec / substeps * 1e3,
                                    "ms_policy_step": sec * 1e3, "policy_steps_per_s": sps,
                                    "hours_for_target": hours, "target": args.target})
            del run
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    if args.hold_s > 0:
        print(f"\nquality: hold the stand pose for {args.hold_s:.1f} s, batch {args.quality_batch}, "
              f"servo gain x{args.gain:g}")
        for dtype in dtypes:
            run = build(ph, args.quality_batch, dtype, args.device, args.timestep, args.gain)
            q = hold_quality(run, args.hold_s, substeps)
            print(f"  {str(dtype)[6:]:>8}: trunk z={q['trunk_z']:.4f} m  max tilt={q['tilt_deg']:.2f} deg  "
                  f"ledger residual={q['residual_J']:.2e} J (supplied {q['supplied_J']:.2f} J)  "
                  f"finite={q['finite']}")
            report["quality"].append({"dtype": str(dtype)[6:], "gain": args.gain, "seconds": args.hold_s, **q})

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
