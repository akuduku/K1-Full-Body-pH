r"""rsl-rl checkpoint -> the firmware's policy file (EigenMLPHist .bin), and a reader + numpy forward pass for tests.

Byte format (RL_Core rl_core/export/policies.py export_eigen_bin, read by K1_Deploy firmware src/policy_mlp.cpp):
per matrix  int32 name_len | name | int32 rows | int32 cols | float64 row-major data, for
actor.0/2/4/6.weight/bias (512-256-128-22, ELU between) and obs_normalizer._mean / _std; the firmware applies
(x - mean) / (std + 0.01), the same as rsl-rl's EmpiricalNormalization.

    python rl/stand/export.py logs/<run>/model_3000.pt logs/<run>/deploy.bin
    # on the robot:  k1_motion_test --task stand --policy deploy.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from . import contract as C
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rl.stand import contract as C


def actor_tensors(checkpoint: str | Path | dict):
    """-> ({actor.N.weight/bias}, mean, std, has_norm) from an rsl-rl (>= 5.5) checkpoint (path or loaded dict)."""
    ck = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck["actor_state_dict"]
    ids = sorted({int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")})
    tensors = {}
    for i in ids:
        tensors[f"actor.{i}.weight"] = state[f"mlp.{i}.weight"].detach().cpu().double()
        tensors[f"actor.{i}.bias"] = state[f"mlp.{i}.bias"].detach().cpu().double()
    has_norm = "obs_normalizer._mean" in state
    obs_dim = tensors[f"actor.{ids[0]}.weight"].shape[1]
    if has_norm:
        mean = state["obs_normalizer._mean"].detach().cpu().double().reshape(-1)
        std = state["obs_normalizer._std"].detach().cpu().double().reshape(-1)
    else:  # no normalizer: mean 0, std 0.99 makes the firmware's (x - mean) / (std + 0.01) the identity
        mean, std = torch.zeros(obs_dim, dtype=torch.float64), torch.full((obs_dim,), 0.99, dtype=torch.float64)
    return tensors, mean, std, has_norm


def check_shapes(tensors) -> None:
    """The firmware loads only the standard 512-256-128 stack on a 72-dim stand observation."""
    want = {"actor.0.weight": (C.HIDDEN_DIMS[0], C.OBS_DIM), "actor.2.weight": (C.HIDDEN_DIMS[1], C.HIDDEN_DIMS[0]),
            "actor.4.weight": (C.HIDDEN_DIMS[2], C.HIDDEN_DIMS[1]), "actor.6.weight": (C.NUM_ACTIONS, C.HIDDEN_DIMS[2])}
    for k, shape in want.items():
        if k not in tensors or tuple(tensors[k].shape) != shape:
            got = tuple(tensors[k].shape) if k in tensors else None
            raise ValueError(f"{k}: {got}, the firmware stand agent needs {shape}")
    if len(tensors) != 8:
        raise ValueError(f"expected 4 linear layers, got {sorted(tensors)}")


def export_bin(checkpoint, out_path: str | Path) -> Path:
    tensors, mean, std, _ = actor_tensors(checkpoint)
    check_shapes(tensors)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write(f, name, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        nb = name.encode()
        f.write(struct.pack("<i", len(nb)))
        f.write(nb)
        f.write(struct.pack("<ii", arr.shape[0], arr.shape[1]))
        f.write(np.ascontiguousarray(arr).tobytes(order="C"))

    with open(out_path, "wb") as f:
        for name, t in tensors.items():
            write(f, name, t.numpy())
        write(f, "obs_normalizer._mean", mean.numpy())
        write(f, "obs_normalizer._std", std.numpy())
    return out_path


def read_bin(path) -> dict[str, np.ndarray]:
    data = Path(path).read_bytes()
    out, pos = {}, 0
    while pos < len(data):
        (n,) = struct.unpack_from("<i", data, pos)
        pos += 4
        name = data[pos:pos + n].decode()
        pos += n
        rows, cols = struct.unpack_from("<ii", data, pos)
        pos += 8
        out[name] = np.frombuffer(data, dtype=np.float64, count=rows * cols, offset=pos).reshape(rows, cols).copy()
        pos += 8 * rows * cols
    return out


def bin_policy(mats: dict[str, np.ndarray]):
    """The firmware's forward pass (policy_mlp.cpp) in numpy: obs (n,72) -> actions (n,22)."""
    mean, std = mats["obs_normalizer._mean"].reshape(-1), mats["obs_normalizer._std"].reshape(-1)

    def act(obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - mean) / (std + C.NORMALIZER_EPS)
        for i in (0, 2, 4, 6):
            x = x @ mats[f"actor.{i}.weight"].T + mats[f"actor.{i}.bias"].reshape(-1)
            if i != 6:
                x = np.where(x >= 0, x, np.expm1(x))
        return x

    return act


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out", nargs="?", default=None, help="default: deploy_<iteration>.bin next to the checkpoint")
    a = ap.parse_args()
    ck = Path(a.checkpoint)
    out = Path(a.out) if a.out else ck.with_name(ck.stem.replace("model", "deploy") + ".bin")
    print(f"wrote {export_bin(ck, out)}")


if __name__ == "__main__":
    main()
