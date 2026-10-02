---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 4.0.0
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Installation

Claveles has been tested in the following environments:

- **Python**: 3.10, 3.11, 3.12, 3.13
- **OS**:
  - Linux (Ubuntu 24.04 LTS recommended; WSL2 on Windows 11 is also supported)
  - Windows 11
  - macOS (Apple silicon, ARM64)

Before proceeding, check that your Python is **3.10 or higher**:

```sh
python3 --version
```

## Install

```sh
python3 -m pip install claveles
```

`claveles` is a meta-package: it installs `claveles-core`, the SDK itself, together with
`claveles-simulator`, the local simulator that {py:class}`~claveles.client.SimulatorClient` runs on. That
is everything this documentation needs.

If you do not want the simulator — when the circuits only ever go to the machine, for instance — install
the core package on its own instead:

```sh
python3 -m pip install claveles-core
```

Building circuits and submitting them to the machine works either way; what a simulator-less install
cannot do is run {py:class}`~claveles.client.SimulatorClient`.

## Extras

`claveles-core` carries one extra, `dev`, which pulls in the test and documentation tooling. It is for
working on the SDK rather than using it:

```sh
python3 -m pip install "claveles-core[dev]"
```

`dev` does not pull in `claveles-simulator`, so install `claveles` as well if the simulator is needed.

## Verify the installation

```{code-cell} python
>>> import claveles
>>> claveles.__version__
```

Next: the [Quickstart](quickstart.ipynb) builds a circuit and runs it on the simulator.
