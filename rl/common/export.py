r"""rsl-rl checkpoint -> the firmware's policy file (EigenMLPHist .bin), and a reader + numpy forward pass for tests.

Byte format (RL_Core rl_core/export/policies.py export_eigen_bin, read by K1_Deploy firmware src/policy_mlp.cpp):
per matrix  int32 name_len | name | int32 rows | int32 cols | float64 row-major data, for
actor.0/2/4/6.weight/bias (512-256-128-<actions>, ELU between) and obs_normalizer._mean / _std; the firmware
applies (x - mean) / (std + 0.01), the same as rsl-rl's EmpiricalNormalization.  The input and output sizes are
the agent's (rl/stand: 72 -> 22, rl/getup: 117 -> 66).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

NORMALIZER_EPS = 0.01  # kNormalizerEps (policy_mlp.cpp)
HIDDEN_DIMS = (512, 256, 128)


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


def check_shapes(tensors, obs_dim: int, num_actions: int, hidden_dims=HIDDEN_DIMS, agent: str = "the") -> None:
    """The firmware loads only the standard 4-layer stack, with the agent's input and output sizes."""
    h = hidden_dims
    want = {"actor.0.weight": (h[0], obs_dim), "actor.2.weight": (h[1], h[0]),
            "actor.4.weight": (h[2], h[1]), "actor.6.weight": (num_actions, h[2])}
    for k, shape in want.items():
        if k not in tensors or tuple(tensors[k].shape) != shape:
            got = tuple(tensors[k].shape) if k in tensors else None
            raise ValueError(f"{k}: {got}, {agent} firmware agent needs {shape}")
    if len(tensors) != 8:
        raise ValueError(f"expected 4 linear layers, got {sorted(tensors)}")


def write_bin(out_path: str | Path, tensors: dict, mean: torch.Tensor, std: torch.Tensor) -> Path:
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


def export_bin(checkpoint, out_path: str | Path, obs_dim: int, num_actions: int, hidden_dims=HIDDEN_DIMS,
               agent: str = "the") -> Path:
    tensors, mean, std, _ = actor_tensors(checkpoint)
    check_shapes(tensors, obs_dim, num_actions, hidden_dims, agent)
    return write_bin(out_path, tensors, mean, std)


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


def bin_policy(mats: dict[str, np.ndarray], eps: float = NORMALIZER_EPS):
    """The firmware's forward pass (policy_mlp.cpp) in numpy: obs (n, obs_dim) -> actions (n, num_actions)."""
    mean, std = mats["obs_normalizer._mean"].reshape(-1), mats["obs_normalizer._std"].reshape(-1)

    def act(obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - mean) / (std + eps)
        for i in (0, 2, 4, 6):
            x = x @ mats[f"actor.{i}.weight"].T + mats[f"actor.{i}.bias"].reshape(-1)
            if i != 6:
                x = np.where(x >= 0, x, np.expm1(x))
        return x

    return act


def load_policy(path: str | Path):
    """A policy file the robot would load (.bin) or an rsl-rl checkpoint (.pt), as the firmware's forward pass."""
    p = Path(path)
    if p.suffix == ".bin":
        return bin_policy(read_bin(p))
    tensors, mean, std, _ = actor_tensors(p)
    mats = {k: v.numpy() for k, v in tensors.items()}
    mats["obs_normalizer._mean"], mats["obs_normalizer._std"] = mean.numpy()[None], std.numpy()[None]
    return bin_policy(mats)
