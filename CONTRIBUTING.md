# Contributing to TensorLens

Thanks for taking the time to contribute. TensorLens is a graduate-level
diagnostics toolkit, so the bar for code quality is intentionally high.

## Development setup

```bash
git clone https://github.com/HAYDARKILIC/tensorlens.git
cd tensorlens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Local quality gates

Before opening a pull request, all of these must pass on your machine:

```bash
make lint        # ruff + black --check
make typecheck   # mypy --strict src/
make test        # pytest with --cov-fail-under=85
make all         # the union of the above
```

CI runs the same commands on Python 3.11 and 3.12.

## Coding standards

* **`from __future__ import annotations`** at the top of every module.
* **PEP 604 union syntax** (`int | None`, not `Optional[int]`).
* **Numpy-style docstrings** with Parameters / Returns / Raises sections.
* **Logging, not printing.** `_log = get_logger(__name__)` at module top.
* **Typed exceptions** — raise from the `TensorLensError` hierarchy.
* **No high-level wrappers** for algorithms — implement from PyTorch /
  NumPy / SciPy primitives.

## Pull request checklist

- [ ] New code is covered by tests (overall coverage ≥ 85%).
- [ ] Docstrings include the math behind any new diagnostic.
- [ ] If you added a notebook, it runs CPU-only in under 60 s.
- [ ] References to peer-reviewed sources are included for any new metric.

## Filing issues

* For bug reports, include a minimal reproducer and the output of
  `python -c "import sys, torch; print(sys.version, torch.__version__)"`.
* For new diagnostics, link to the paper that motivates them.
