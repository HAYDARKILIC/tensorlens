# TensorLens — Architecture Overview

This document describes the internal organization of TensorLens, the design
principles that govern its modules, and how the curriculum maps onto the
codebase.

## 1. Design principles

1. **First-principles only.** Every algorithm is implemented from PyTorch /
   NumPy / SciPy primitives. No `sklearn`, no `umap-learn`, no
   `tensorboard`, no `wandb`, no `pytorch-hessian`. The point is to expose
   the math, not hide it.
2. **Diagnostic, not training.** TensorLens is a *spectroscope* on a
   pre-trained or in-training model — never the trainer itself. The
   `BitWise-LLM-Forge` repository handles training.
3. **Strict typing.** `from __future__ import annotations` everywhere; PEP
   604 union syntax; `mypy --strict` clean.
4. **Reproducibility-first.** Every stochastic function takes a `seed`
   argument and uses a `torch.Generator` — no implicit reliance on global
   RNG state.
5. **Structured logging.** No `print` calls in `src/`. Use
   `src.utils.get_logger(__name__)` and set verbosity via the
   `TENSORLENS_LOG_LEVEL` environment variable.
6. **Custom exception hierarchy.** All raised exceptions inherit from
   `TensorLensError`. Library callers catch *one* type.

## 2. Package map

```
src/
├── utils/                 Shared infrastructure
│   ├── __init__.py        Logging, seeding, exception hierarchy
│   └── synthetic.py       Mathematically-realistic test fixtures
│
├── geometry/              Week 1 — High-dim manifold geometry
│   ├── anisotropy.py             Cone-collapse metrics, SVD diagnostics
│   ├── manifold_projection.py    From-scratch t-SNE and UMAP
│   └── intrinsic_dimension.py    TwoNN + MLE estimators
│
├── mechanistic/           Weeks 2 & 5 — Residual stream + SAE
│   ├── residual_stream_hooks.py  Forward-hook capture engine
│   ├── attention_entropy.py      Head archetype classification
│   ├── sparse_autoencoder.py     Overcomplete SAE
│   └── feature_graph.py          Jaccard co-occurrence graph
│
├── optimization/          Week 3 — Loss landscape topology
│   ├── filter_normalization.py   Li et al. 2018 direction generator
│   ├── hessian_power_iteration.py HVP + power iteration + Lanczos
│   ├── trajectory_capture.py     Snapshot store, loss grid
│   └── edge_of_stability.py      EOS detector
│
└── profiler/              Week 4 — HPC roofline analysis
    ├── memory_hierarchy.py       A100 / H100 published parameters
    ├── log_parser.py             Nsight JSON / CSV parsers
    ├── roofline.py               Multi-tier roofline model
    └── kernel_gantt.py           Async pipeline visualization
```

## 3. Curriculum-to-code mapping

| Week | Notebook                                       | Source modules                                                                  |
| ---- | ---------------------------------------------- | -------------------------------------------------------------------------------- |
| 1    | `01_representation_geometry.ipynb`             | `geometry.anisotropy`, `geometry.manifold_projection`, `geometry.intrinsic_dimension` |
| 2    | `02_residual_stream_flow.ipynb`                | `mechanistic.residual_stream_hooks`, `mechanistic.attention_entropy`             |
| 3    | `03_loss_landscape_trajectory.ipynb`           | `optimization.*`                                                                 |
| 4    | `04_hardware_roofline_profiler.ipynb`          | `profiler.*`                                                                     |
| 5    | `05_mechanistic_interpretability.ipynb`        | `mechanistic.sparse_autoencoder`, `mechanistic.feature_graph`                    |
| 6    | `06_capstone_webgl_dashboard.py`               | `utils.*` (read-only consumer)                                                   |

## 4. Notebook conventions

Every notebook follows the same structure:

1. **Section 1 — Theory.** Closed-form math, references, intuition.
2. **Section 2 — Setup.** Imports + reproducibility seeding.
3. **Sections 3+ — Compute.** Calls into `src/` modules with synthetic data.
4. **Final section — Take-aways.** Bullet-point summary, references.

Synthetic data comes from `src.utils.synthetic`. Notebooks do not require
any pretrained model download; everything runs CPU-only in under 60 s.

## 5. Testing & quality gates

* `pytest tests/` — 90+ unit tests across all modules.
* `pytest --cov=src --cov-report=term --cov-fail-under=85` — coverage gate.
* `ruff check src/ tests/` — lint.
* `black --check src/ tests/` — format.
* `mypy --strict src/` — type check.

The CI workflow in `.github/workflows/ci.yml` runs all five gates on Python
3.11 and 3.12.

## 6. The capstone

`notebooks/06_capstone_webgl_dashboard.py` is a self-contained `aiohttp`
WebSocket server. The producer task generates training metrics; each
connected client gets its own `asyncio.Queue` (fan-out, bounded). The wire
protocol is MessagePack to keep frame size small and parsing fast in the
browser. The front-end is a single embedded HTML page using Plotly.js
`scattergl` for GPU-accelerated line rendering.

To swap the simulator for real training, replace `SimulatedTrainer._next()`
with a function that returns a `TrainingMetric` constructed from your
actual training loop's state.
