# The K1 pH plant in C++/CUDA

`ph_k1/plant.py` (`K1PHPlant`) written again in C++, so that RL can train on it: the same port-Hamiltonian
model (free trunk, 22 hinges, 128 contact points with bristles and the Coulomb cone, 22 DC motors, volts in),
the same integrator, the same numbers. The Python plant stays the reference.

- **One CUDA thread = one robot.** The physics is header-only C++ (`include/ph_k1/`) that compiles with
  g++ (CPU, OpenMP over robots) and with nvcc (GPU) from the same source, in float32 or float64.
- **One call per policy step.** `policy_step(targets)` runs the servo of `ph_k1/drive.py` (target angles
  → volts) and the 20 plant steps of a 50 Hz policy step inside one C++ call / one kernel launch.
- **State in PyTorch tensors**, updated in place (zero copy): `q (B,29)`, `p (B,28)`, `xi (B,128,2)`,
  `lam (B,22)`, the ledger `(B,6)`.
- **No MJCF parsing in C++**: `ph_k1/model.py` reads the XML; `ph_k1/native.py` sends the numbers once.
- Cheaper tree algorithms with the same results: the mass matrix by composite rigid bodies, generalised
  forces by one backward pass of body wrenches, contact Jacobians only for points in contact, no host syncs
  (the contact mode switch is decided per robot).

## Build

```bash
cd "Physik C++" && python setup.py build_ext --inplace       # -> ph_k1_native*.so next to setup.py
```

CUDA is built when nvcc exists (Kaggle, Colab), CPU-only otherwise (this laptop). `PH_K1_CUDA=1` requires
CUDA, `PH_K1_CUDA=0` forces CPU. Architectures: `TORCH_CUDA_ARCH_LIST` (default `7.5;8.0;8.6;8.9`).

## Use

```python
from ph_k1.model import load_k1
from ph_k1.native import K1PHPlantNative

plant = K1PHPlantNative(model=load_k1(dtype=torch.float32, device="cuda"), batch=4096)
plant.reset()
info = plant.policy_step(targets)        # servo + 20 plant steps, targets (B,22) joint angles
info = plant.step(volts)                 # or one plant step with volts (B,22)
info.extra["nu"]                         # nu_{n+1}: trunk twist + joint speeds
```

`K1PHPlantNative` is a subclass of `K1PHPlant`: `reset`, `kinematics`, `velocity`, `energy`, `ledger`,
`ph_matrices` work as before. `python checks.py --impl native` runs the whole validation suite with it.

## Tests and results

```bash
python -m pytest "Physik C++/tests" -q                       # C++ vs Python, float64, CPU
PH_TEST_DEVICE=cuda python -m pytest "Physik C++/tests" -q   # the same on a GPU
python checks.py --impl native                               # 9/9 PASS
python benchmarks/benchmark_ph.py --impl native              # speed
```

One step with stick, slip and lift-off agrees with Python to ~1e-13; 300 steps of a drop, the welded trunk
with end stops, flight, saturated motors and the servo to ~1e-8; float32 holds the stand pose with the same
energy-ledger residual (0.015 J).

Laptop CPU (6 threads), float32, policy steps per second (50 Hz policy = 20 plant steps):

| robots | PyTorch plant | C++ plant |
|---:|---:|---:|
| 1 | 9 | 1,430 |
| 256 | 195 | 9,390 |
| 1024 | 222 | 7,970 |

On the GPU: `kaggle/run.sh` (Kaggle) or `colab/ph_bench_colab.ipynb` build the CUDA version, run the tests
on the GPU, benchmark it next to the PyTorch plant and MJX, and upload a video of the servo hold + push to W&B.
