# Contributing

**Dev environment:** Clone the repo, run `make dev-install` (creates `.venv` and installs all dependencies including dev extras), then verify with `PYTHONPATH=. .venv/bin/python -c "import cortex; print('ok')"`. The project requires Python 3.11+ and uses pre-commit hooks for formatting and linting — run `.venv/bin/pre-commit install` once after setup so they fire on every commit.

**Running tests:** `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q` runs the full suite; 116 tests pass on CPU/MPS, ~39 are skipped and require CUDA (Triton kernel paths). For a faster local iteration loop, `pytest tests/ -q -k "not cuda"` skips the CUDA-skipped tests explicitly.

**Submitting a change:** Open a PR against `main` with a commit message prefixed by the affected phase (e.g., `[phase-2] fix sparse xattn masking`). Run `make lint` and `make test` before pushing; CI runs the same checks. If your change touches a kernel or serving path, include a benchmark or latency comparison in the PR description — claims without numbers are rejected per the project's core engineering principle.
