# TensorLens

**First-principles diagnostic toolkit for modern LLM mechanics, high-dimensional geometry, and mechanistic interpretability.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage 92%](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](#)
[![Type Checked: mypy --strict](https://img.shields.io/badge/mypy-strict-2a6db2.svg)](#)

---

Every algorithm — SVD anisotropy metrics, t-SNE, UMAP, Hessian power iteration, Lanczos, sparse autoencoders, multi-tier roofline — is implemented from PyTorch/NumPy/SciPy primitives. No `sklearn`, no `umap-learn`, no `tensorboard`, no opaque wrappers.

Companion to [**BitWise-LLM-Forge**](#) (the training-side repository): where BitWise constructs and trains transformers from bit-level numerics, TensorLens dissects what they *actually do* in their latent geometries.

## Curriculum

| Week | Topic                                  | Theory                                                                                            | Notebook                                       |
| ---- | -------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| 1    | Representation geometry                | Anisotropy index, effective rank, t-SNE/UMAP topology                                             | `01_representation_geometry.ipynb`             |
| 2    | Residual stream & information flow     | Mechanistic framework, attention entropy, induction heads                                         | `02_residual_stream_flow.ipynb`                |
| 3    | Loss landscapes & Edge of Stability    | Filter-normalized directions, Hessian power iteration, $\lambda_\max \geq 2/\eta$                 | `03_loss_landscape_trajectory.ipynb`           |
| 4    | HPC roofline & memory hierarchy        | Arithmetic intensity, multi-tier ceilings, kernel Gantt                                           | `04_hardware_roofline_profiler.ipynb`          |
| 5    | Polysemanticity & sparse autoencoders  | Superposition hypothesis, monosemantic dictionary learning                                        | `05_mechanistic_interpretability.ipynb`        |
| 6    | Capstone — WebGL live dashboard        | Async telemetry, MessagePack over WebSocket, GPU-side rendering                                   | `06_capstone_webgl_dashboard.py`               |

## Key features

- **Zero black boxes.** SVD, KL gradient descent for t-SNE, fuzzy-simplicial-set UMAP, double-backward HVP, Lanczos with full re-orthogonalization, SAE with decoder unit-norm projection — all from scratch.
- **Mathematical rigor.** Every module's docstring cites the originating paper and reproduces the implemented equation in LaTeX.
- **Production grade.** `mypy --strict`, `ruff`, `black`, 91 unit tests, 92% coverage, CI on Python 3.11 & 3.12.
- **Reproducibility-first.** Deterministic seeding, snapshot-based weight capture, custom exception hierarchy.
- **Realistic synthetics.** Test fixtures replicate empirically-observed phenomena: cone-collapsed embeddings, attention sinks, induction heads, polysemantic codes, EOS oscillations.

## Repository structure

```
tensorlens/
├── src/
│   ├── geometry/         # Week 1 — anisotropy, t-SNE, UMAP, intrinsic dim
│   ├── mechanistic/      # Weeks 2 & 5 — hooks, attention, SAE, feature graph
│   ├── optimization/     # Week 3 — filter normalization, Hessian, EOS
│   ├── profiler/         # Week 4 — Nsight/CSV parsers, roofline, Gantt
│   └── utils/            # shared logging, seeding, synthetic fixtures
├── notebooks/            # six numbered curriculum notebooks
├── tests/                # 91 tests, 92% coverage
├── docs/                 # architecture + mathematical appendix
├── .github/workflows/    # CI: lint + typecheck + test matrix
├── pyproject.toml
├── Makefile
└── README.md
```

## Installation

```bash
git clone https://github.com/HAYDARKILIC/tensorlens.git
cd tensorlens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### Quick example — anisotropy diagnostic

```python
import torch
from src.utils.synthetic import AnisotropicConfig, synth_anisotropic_hidden_states
from src.geometry.anisotropy import full_report

# Synthesize cone-collapsed embeddings (or load your own LLM hidden states).
h = synth_anisotropic_hidden_states(AnisotropicConfig(n_tokens=1024, dim=768, anisotropy=0.7))

report = full_report(h, sample_size=512)
print(report)
# AnisotropyReport(N=1024, d=768, anisotropy=0.6182, r_eff=18.43, gini=0.7421)
```

### Quick example — Hessian top eigenvalue

```python
from src.optimization import hessian_top_eigenvalue, lanczos_extrema

lam, _ = hessian_top_eigenvalue(loss_fn, model, n_iter=30)
lam_min, lam_max = lanczos_extrema(loss_fn, model, m=20)
```

### Run the notebooks

```bash
jupyter lab notebooks/
```

### Capstone live dashboard

```bash
python notebooks/06_capstone_webgl_dashboard.py --port 8765
# Then open http://localhost:8765
```

## Development

```bash
make lint        # ruff + black --check
make typecheck   # mypy --strict src/
make test        # pytest with coverage
make all         # all of the above
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — module map and design principles
- [`docs/mathematical_appendix.md`](docs/mathematical_appendix.md) — closed-form derivations for every diagnostic
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — quality gates and review checklist

## References

The implementation draws directly on these papers; each module cites the originating source in its docstring.

- Ethayarajh (2019). *How Contextual are Contextualized Word Representations?* EMNLP.
- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.* Anthropic.
- Olsson et al. (2022). *In-context Learning and Induction Heads.* Anthropic.
- Bricken et al. (2023). *Towards Monosemanticity.* Anthropic.
- van der Maaten & Hinton (2008). *Visualizing Data using t-SNE.* JMLR.
- McInnes, Healy, Melville (2018). *UMAP.* arXiv:1802.03426.
- Facco et al. (2017). *Estimating the intrinsic dimension of datasets.* Sci. Reports.
- Li et al. (2018). *Visualizing the Loss Landscape of Neural Nets.* NeurIPS.
- Cohen et al. (2021). *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR.
- Williams, Waterman, Patterson (2009). *Roofline.* CACM.

## License

MIT — see [`LICENSE`](LICENSE).
