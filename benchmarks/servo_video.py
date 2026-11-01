"""Video of the C++/CUDA pH plant: the classical servo holds the stand pose, a sideways push at 2 s.

Proves on the training machine that the native plant, headless rendering and the W&B upload work,
before any RL runs on it.  Like hold_pose.py, but every policy step is one K1PHPlantNative.policy_step.

    MUJOCO_GL=egl python benchmarks/servo_video.py --device cuda --out outputs/videos/hold_push.mp4
    WANDB_API_KEY=... python benchmarks/servo_video.py --device cuda --wandb-project "Akos pH"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ph_k1.model import load_k1  # noqa: E402
from ph_k1.native import K1PHPlantNative  # noqa: E402
from ph_k1.plant import Ledger  # noqa: E402
from ph_k1.video import VideoRecorder, log_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--push", type=float, default=20.0, help="sideways trunk speed change at 2 s [cm/s]")
    ap.add_argument("--gain", type=float, default=3.0, help="servo gain scale (1 = firmware gains, tips over)")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "videos" / "hold_push.mp4")
    ap.add_argument("--wandb-project", default="", help="log the video to this W&B project (needs WANDB_API_KEY)")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    ap.add_argument("--wandb-name", default="native-servo-hold")
    ap.add_argument("--summary", type=Path, default=None, help="benchmark summary.json to attach to the W&B run")
    args = ap.parse_args()

    model = load_k1(dtype=getattr(torch, args.dtype), device=args.device)
    plant = K1PHPlantNative(model=model, batch=1)
    plant.reset()
    target = plant.theta.clone()
    rec = VideoRecorder(plant)
    steps = int(round(args.seconds / 0.02))
    log = []
    wall = time.time()
    for it in range(steps):
        if args.push and it == int(round(2.0 / 0.02)):
            # impulse on the trunk: change its sideways momentum (world y); the ledger restarts (hold_pose.py)
            kin = plant.kinematics()
            nu = plant.velocity(kin)
            dv = torch.tensor([0.0, args.push / 100, 0.0], dtype=nu.dtype, device=nu.device)
            nu[:, 3:6] += kin.R[:, 0].transpose(1, 2) @ dv
            plant.p = (kin.M @ nu[..., None]).squeeze(-1)
            plant.ledger = Ledger.zeros(plant.B, plant.dtype, plant.device)
        info = plant.policy_step(target, gain_scale=args.gain)
        rec.capture()
        q = plant.q[0]
        tilt = float(torch.rad2deg(torch.arccos((1 - 2 * (q[4] ** 2 + q[5] ** 2)).clamp(-1, 1))))
        log.append({"t": round((it + 1) * 0.02, 3), "trunk_z": float(q[2]), "tilt_deg": tilt,
                    "contacts": int(info.contact_active.sum()), "supplied_J": float(plant.ledger.supplied[0])})
    path = rec.save(args.out)
    last = log[-1]
    upright = last["tilt_deg"] < 10 and last["trunk_z"] > 0.4
    print(f"{'UPRIGHT' if upright else 'FELL'} after {args.seconds:.1f} s (push {args.push:g} cm/s at 2 s): "
          f"trunk z={last['trunk_z']:.3f} m, tilt={last['tilt_deg']:.2f} deg; video {path} "
          f"({len(rec.frames)} frames, {time.time() - wall:.1f} s wall)")
    (path.with_suffix(".json")).write_text(json.dumps(log, indent=1) + "\n")

    if args.wandb_project:
        import wandb

        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name,
                         job_type="validation", config={"device": args.device, "dtype": args.dtype,
                                                        "push_cm_s": args.push, "gain": args.gain})
        for row in log:
            wandb.log({k: v for k, v in row.items() if k != "t"}, step=int(round(row["t"] / 0.02)))
        log_video(path, "video/servo_hold_push", step=steps)
        if args.summary and args.summary.exists():
            art = wandb.Artifact("ph-benchmark", type="benchmark")
            art.add_file(str(args.summary))
            run.log_artifact(art)
        print(f"W&B: {run.url}")
        run.finish()
    return 0 if upright else 1


if __name__ == "__main__":
    sys.exit(main())
