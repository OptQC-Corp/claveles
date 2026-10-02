(claveles-index)=

# Claveles Documentation

Claveles is a Python library for continuous-variable, measurement-based optical quantum computing. You build
a circuit out of modes and operations, hand it to a client, and read the measured values — and, on the
simulator, the quantum state the circuit left behind — back out.

The same circuit runs in two places, and which one you have access to decides where to start:

* **On the local simulator.** No credentials and no network, just the default installation. It also holds
  states the machine cannot prepare, such as Fock, cat and GKP states, and can apply
  non-Gaussian gates. Start with [Installation](installation.md) and the
  [Quickstart](quickstart.ipynb), whose cells are executed when this documentation is built.
* **On the machine**, an optical quantum computer that performs measurement-based quantum computing on a
  two-dimensional cluster state. It needs an access token; [Executing Circuit](execution.ipynb) picks up
  where the quickstart leaves off.

A program is expressed in one of three representations — **circuit**, **graph** or **machinery** — which
sit at descending levels of abstraction. Writing the circuit representation and letting the client
compile it is the normal path; the [Quickstart](quickstart.ipynb) introduces all three in a page.

## Get started

* [Installation](installation.md) — supported environments, and which package to install.
* [Quickstart](quickstart.ipynb) — build one circuit, run it on the simulator, and read the result.

The quickstart speaks of modes, quadratures and homodyne measurement. If continuous-variable quantum
computing is new to you, read [Introduction to Quantum States](states.md),
[Introduction to Measurement](measurement.md) and [Introduction to Gate Operations](ops.md) first: they
start from what a mode is, follow on from each other, and need nothing from the SDK.

```{toctree}
:caption: Get started
:hidden:
installation.md
quickstart.ipynb
```

## Building a program

* [Constructing Circuit Representation](circuit_repr.ipynb) — modes, gates, measurement, feedforward,
  composing and editing circuits, saving and loading.
* [Working with Quantum States](quantum_states.md) — building a state object, reading it, placing it on a
  circuit and reading what a simulation gives back.
* [Feedforward](feedforward.ipynb) — making an operation's parameter depend on a measurement result.
* [Constructing Graph Representation](graph_repr.ipynb) — placing operations on the macronode lattice
  yourself, and packing several shots into one graph.
* [Constructing Machinery Representation](machinery_repr.ipynb) — driving the machine at the level of
  measurement angles.

```{toctree}
:caption: Building a program
:hidden:
circuit_repr.ipynb
quantum_states.md
feedforward.ipynb
graph_repr.ipynb
machinery_repr.ipynb
```

## Running a program

* [Clients](clients.md) — what a client is, how the two of them differ, and the errors they raise.
* [Simulating Circuit](simulation.ipynb) — the simulator's options, the state a shot leaves behind, and
  Wigner functions.
* [Executing Circuit](execution.ipynb) — submitting to the machine, and what the result holds.
* [Executing Circuit Asynchronously](async_execution.ipynb) — submitting a job and collecting it later.
* [OptQC Client](optqc.md) — the machine client's parameters, compilation settings and methods.

```{toctree}
:caption: Running a program
:hidden:
clients.md
simulation.ipynb
execution.ipynb
async_execution.ipynb
optqc.md
```

## Visualization

* [Visualizing Circuit Representation](viz_circuit_repr.ipynb)
* [Visualizing Graph Representation](viz_graph_repr.ipynb)
* [Visualizing Machinery Representation](viz_machinery_repr.ipynb)

```{toctree}
:caption: Visualization
:hidden:
viz_circuit_repr.ipynb
viz_graph_repr.ipynb
viz_machinery_repr.ipynb
```

## Background

Three introductions to continuous-variable quantum computing, readable in order and independent of the
SDK.

* [Introduction to Quantum States](states.md) — what a mode is, and the states that appear throughout.
* [Introduction to Measurement](measurement.md) — homodyne and Fock-number measurement, and why they are
  complementary.
* [Introduction to Gate Operations](ops.md) — Gaussian and non-Gaussian operations, and what makes a
  gate set universal.

```{toctree}
:caption: Background
:hidden:
states.md
measurement.md
ops.md
```

## Primitives

What a program is built out of, enumerated: every state, every operation and every measurement the SDK
offers, each with the introduction to the physics behind it. These are pages to look something up in
rather than to read through.

* [States](primitives/states.md) — every state factory, and the representations a state can be held in.
* [Gates & Operations](primitives/gates.md) — every operation, with its definition, its Heisenberg action
  and where it can run.
* [Measurements](primitives/measurements.md) — homodyne and Fock-number measurement as operations you can
  place.

```{toctree}
:caption: Primitives
:hidden:
:maxdepth: 2
primitives/states.md
primitives/gates.md
primitives/measurements.md
```

## Reference

* [Support Matrix](support.md) — which operations, states and backends can be used where, generated from
  the table the SDK itself consults.
* [API Reference](reference/modules.rst) — the modules, classes and functions.

```{toctree}
:caption: Reference
:hidden:
:maxdepth: 2
support.md
reference/modules.rst
```

## License

Claveles (`claveles` and `claveles-core`) is **free** and **open-source** software released under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The standard installation of Claveles (`pip install claveles`) automatically includes `claveles-simulator` as a default simulation backend. While `claveles` is open-source, `claveles-simulator` is maintained in a separate repository and distributed as a pre-compiled binary wheel under its own proprietary End User License Agreement (EULA).

* [Claveles Simulator License](simulator_license_and_third-party.md) — EULA terms and separate third-party open-source notices for the simulator backend.

```{toctree}
:caption: License
:hidden:
simulator_license_and_third-party.md
```
