# Claveles

**Claveles is a software development kit for continuous-variable quantum computers.**
It helps you design quantum circuits, simulate the dynamics, and execute programs on real hardware.

> Looking for end‑user docs? See the user guide at [docs/source/index.md](docs/source/index.md).

## Packages

This repository produces two PyPI packages:

| Package | Description | Install name |
| ------- | ----------- | ------------ |
| **`claveles`** | Meta-package — depends on `claveles-core` and `claveles-simulator` | `pip install claveles` |
| **`claveles-core`** | SDK source — the full library in this repo | `pip install claveles-core` |

A third package, **`claveles-simulator`** (C++ binary), is maintained in a separate repository and distributed as a pre-built wheel.

## Requirements

- **Supported Platforms**
  - Linux (Ubuntu 24.04 LTS recommended; WSL2 on Windows 11 is also supported)
  - Windows 11
  - macOS (Apple silicon, ARM64)
- **Python versions**: 3.10+ (Python 3.14 is expected to work, but some features remain unverified)

## Installation

### Standard Installation (SDK + Simulator)

To install the full package including both `claveles-core` and `claveles-simulator`:

```sh
pip install claveles
```

### Core SDK Only
If you only need the SDK without the local C++ simulator:

```sh
pip install claveles-core
```

### Development Tools ([dev])

To install tools required for testing and documentation:

```sh
pip install "claveles-core[dev]"
# Or from the claveles-core source directory:
# python -m pip install ".[dev]"
```

### Using the C++ Simulator

Since claveles includes claveles-simulator by default, SimulatorClient is available immediately after standard installation:

```python
from claveles.client import SimulatorClient

client = SimulatorClient(n_shots=1)
```

The binding itself is reachable as claveles.simulator (from claveles import simulator), but it only exposes the low-level simulate_* entry points; SimulatorClient is the supported way in.

## Quickstart

After installation, try a minimal program:

```python
from math import pi
from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import QuantumState

# Create a circuit representation of a program
c = CircuitRepr("sample_circuit")
c.Q(0) | QuantumState.squeezed(r=1.0, phi=0.0)  # Allocate a squeezed state as an input
c.Q(0) | std.PhaseRotation(phi=pi / 2)          # Apply a phase rotation of pi/2 to qumode 0
c.Q(0) | std.MeasureHomodyne(phi=pi / 2)        # Measure qumode 0 (homodyne)

print(c)
```

See the user docs for details: [docs/source/index.md](docs/source/index.md).

## Tests

**Some test suites are long‑running. Use `pytest-xdist` to parallelize.**

```sh
# basic tests
pytest

# tests requiring network access
pytest --network

# tests requiring simulator access
pytest --simulator

# long‑running tests (parallel)
pytest -n auto --longrun

# everything
pytest -n auto --longrun --network --simulator
```

## Regenerating protobuf files

The `src/claveles/pb/` directory contains Python code auto-generated from the proto definitions in `proto/`.
Regenerate it when the `.proto` files change.

### Prerequisites

Install [buf](https://buf.build/docs/installation) **exactly v1.69.0** (the version is enforced by `codegen/Makefile`):

```sh
# Linux / macOS (adjust the binary name for your platform)
mkdir -p bin
curl -sSL "https://github.com/bufbuild/buf/releases/download/v1.69.0/buf-$(uname -s)-$(uname -m)" \
  -o bin/buf && chmod +x bin/buf
export PATH="$(pwd)/bin:$PATH"
```

Verify:

```sh
buf --version   # must print: 1.69.0
```

You also need a buf registry token (`BUF_TOKEN`) to pull the remote plugins:

```sh
echo "${BUF_TOKEN}" | buf registry login --token-stdin
```

### Generate

```sh
cd codegen
make clean && make all
```

The generated files are written to `../src/claveles/pb/` (relative to `codegen/`).

## Project layout

```text
├── codegen/
├── docs/
├── examples/
├── meta/
├── proto/
├── src/
│   └── claveles/
│       ├── circuit/
│       ├── client/
│       ├── execute/
│       ├── feedforward/
│       ├── graph/
│       ├── machinery/
│       └── pb/
├── tests/
└── tools/
```

- `codegen/` : `buf` configuration for regenerating `src/claveles/pb/`.
- `docs/` : User & developer documentation sources.
- `meta/` : Packaging for the `claveles` meta-package.
- `proto/` : Protocol Buffers definitions, the input to `codegen/`.
- `src/claveles/` : Main source tree.
  - `circuit/` : Circuit representation.
  - `client/` :  Execution client and result types.
  - `execute/` : Unified wrapper over multiple clients; one-call submit & fetch results.
  - `feedforward/` : Mechanisms to update operation parameters conditioned on measurement outcomes.
  - `graph/` : Graph representation.
  - `machinery/` : Machinery representation.
  - `pb/` : Protocol Buffers (auto-generated).
- `tests/` : Unit and integration tests.
- `tools/` : Helper scripts for CI version patching and documentation generation.

## Contributing

Contributions are very welcome! Whether it's reporting bugs, improving documentation, or submitting code changes, any help is greatly appreciated.

Please check our [Contributing Guide](CONTRIBUTING.md) for instructions on setting up your development environment and submitting pull requests.

> **Note:** All contributors must certify their commits under the [Developer Certificate of Origin (DCO)](DCO) by signing off their commits (`git commit -s`).

## License

Claveles is **free** and **open-source** software, released under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Note that `claveles-simulator` (installed by default as part of the `claveles` meta-package) is hosted as a separate project on PyPI and released under its own proprietary license.

This package includes a fork of [MQC3](https://github.com/HidehiroYonezawa/mqc3), originally developed by RIKEN and licensed under the MIT License. See [NOTICE.md](NOTICE.md) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

Required attributions for algorithms derived from external libraries are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).