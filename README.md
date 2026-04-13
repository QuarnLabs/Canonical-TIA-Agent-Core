# Canonical TIA Agent Core

Reference implementation of a **Canonical TIA Agent Core** for benchmarking **Task AI** vs **Interface AI** in the framework of the Theory of the Abstraction Interface (TIA).

## What is included

- frozen reference vocabulary for the minimal TIA agent loop,
- a pluggable `BaseTIAAgent` abstraction,
- a small benchmark battery for interface-oriented architectural tests,
- mock agents and environments for reproducible demonstrations,
- a dual-license setup: **AGPLv3 + commercial license**.

## Licensing

This repository is available under a **dual-license model**:

- **AGPLv3** for open-source and reciprocal use,
- **Commercial License** for proprietary or closed-source deployments.

See `LICENSE-AGPLv3.md` and `COMMERCIAL_LICENSE.md`.

## Frozen core

The frozen reference core includes:

- state vocabulary: `x_t`, `z_t`, `s_t`, `L_t`, `v_t`, `Phi_t`, `A_t`, `q_t`, `chi_t`, `u_t`,
- one canonical lifecycle loop,
- a standard telemetry schema,
- benchmark hooks for self-relevant perturbation and related tests.

The following remain **plugin/extensible**:

- `encoder`
- `self_model`
- `loss_fn`
- `integration_metric`
- `access_metric`
- `interface_synthesis`
- `policy`
- environment implementation

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

```bash
python scripts/run_mock_benchmark.py
```

```bash
pytest -q
```
