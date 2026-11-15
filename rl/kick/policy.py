r"""A kick policy file as a function of observations to actions (torch, batched): a TorchScript module, an rsl-rl
checkpoint or the firmware's .bin (rl/common/export.py), with the firmware's forward pass (512-256-128 ELU,
(x - mean) / (std + 0.01) in front)."""

from __future__ import annotations

from pathlib import Path

import torch

from rl.common.export import NORMALIZER_EPS, actor_tensors, read_bin


def load(path: str | Path, device="cpu", dtype=torch.float32):
    """-> (policy(obs (B,n)) -> (B,22), obs_dim)."""
    p = Path(path)
    if p.suffix == ".pt":
        try:
            net = torch.jit.load(str(p), map_location=device).eval()
        except RuntimeError:
            net = None
        if net is not None:
            def scripted(obs):
                with torch.no_grad():
                    return net(obs.float()).to(obs.dtype)
            return scripted, None
    if p.suffix == ".bin":
        mats = read_bin(p)
        T = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
        layers = [(T(mats[f"actor.{i}.weight"]), T(mats[f"actor.{i}.bias"]).reshape(-1)) for i in (0, 2, 4, 6)]
        mean, std = T(mats["obs_normalizer._mean"]).reshape(-1), T(mats["obs_normalizer._std"]).reshape(-1)
    else:
        tensors, mean, std, _ = actor_tensors(p)
        ids = sorted({int(k.split(".")[1]) for k in tensors})
        layers = [(tensors[f"actor.{i}.weight"].to(device, dtype), tensors[f"actor.{i}.bias"].to(device, dtype))
                  for i in ids]
        mean, std = mean.to(device, dtype), std.to(device, dtype)

    def policy(obs):
        with torch.no_grad():
            x = (obs.to(dtype) - mean) / (std + NORMALIZER_EPS)
            for k, (w, b) in enumerate(layers):
                x = x @ w.T + b
                if k < len(layers) - 1:
                    x = torch.nn.functional.elu(x)
            return x.to(obs.dtype)

    return policy, int(layers[0][0].shape[1])
