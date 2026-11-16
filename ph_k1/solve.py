"""The plant's linear solves in one place.

torch.linalg.solve everywhere, except: some CUDA builds of PyTorch have no GPU code for batched solves on older GPUs
(Kaggle, torch 2.11+cu130 on a Tesla T4, 16 Nov 2026: "CUDA error: no kernel image is available").  With ON_CPU set,
the systems are solved by LAPACK on the CPU and the result goes back to the GPU; autograd follows through the copy.
The matrices are small (28 x 28 per robot), so the copies cost little next to the rest of a plant step.
"""

from __future__ import annotations

import torch

ON_CPU = False   # set by probe(), or by hand


def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if ON_CPU and A.is_cuda:
        return torch.linalg.solve(A.cpu(), B.cpu()).to(A.device)
    return torch.linalg.solve(A, B)


def probe(device, batch: int = 64, n: int = 28, dtype=torch.float32) -> bool:
    """Try one batched solve on `device`; switch to the CPU path if the GPU has no code for it.  -> ON_CPU."""
    global ON_CPU
    device = torch.device(device)
    if device.type != "cuda" or ON_CPU:
        return ON_CPU
    try:
        A = torch.eye(n, dtype=dtype, device=device).expand(batch, n, n).contiguous()
        torch.linalg.solve(A, torch.ones(batch, n, dtype=dtype, device=device))
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001  (the CUDA error type differs between PyTorch versions)
        ON_CPU = True
        print(f"[ph_k1] batched solve not available on this GPU ({type(exc).__name__}): solving on the CPU", flush=True)
    return ON_CPU
