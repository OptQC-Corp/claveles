(sec:support)=

# Support Matrix

Which operations and states can be used where. This page is generated from {py:mod}`claveles.support`,
the table the SDK itself consults, so it says exactly what the SDK will accept or refuse.

`✓` means supported and `—` means not supported.

```{note}
Do not edit this page by hand: it is written by `tools/gen_support_page.py`, and a test fails when the
two disagree. Change {py:mod}`claveles.support` and re-run that script instead.
```

For what each operation *does*, see {ref}`Gates <sec:gates>`, which gives every one of them with its
definition and its Heisenberg action.

(sec:support-operations)=

## Operations

A **representation** is how a program is expressed: {py:mod}`~claveles.circuit` is what the simulator
runs, while {py:mod}`~claveles.graph` and {py:mod}`~claveles.machinery` are what the machine runs, so an
operation is executable on the machine exactly when it has both of those. A **state representation**
is how the simulator holds the state the operation acts on.

The last column says why an operation is missing from a column; it is empty for the operations that
run everywhere.

| Operation | circuit | graph | machinery | bosonic | fock | Why restricted |
| --- | --- | --- | --- | --- | --- | --- |
| `intrinsic.measurement` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.displacement` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.phase_rotation` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.shear_x_invariant` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.shear_p_invariant` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.squeezing` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.squeezing45` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.arbitrary` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.controlled_z` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.beam_splitter` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.two_mode_shear` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `intrinsic.manual` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.displacement` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.measure_homodyne` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.squeezing` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.p_gate` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.beam_splitter` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.phase_rotation` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.q_gate` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.squeezing45` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.arbitrary` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.controlled_z` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.two_mode_shear` | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `std.kerr` | ✓ | — | — | — | ✓ | Non-Gaussian, so it needs the Fock backend and cannot be compiled for the machine |
| `std.cubic_phase` | ✓ | — | — | — | ✓ | Non-Gaussian, so it needs the Fock backend and cannot be compiled for the machine |
| `std.v_gate` | ✓ | — | — | — | ✓ | Non-Gaussian, so it needs the Fock backend and cannot be compiled for the machine |
| `std.toffoli` | ✓ | — | — | — | ✓ | Non-Gaussian, so it needs the Fock backend and cannot be compiled for the machine |
| `std.ccz` | ✓ | — | — | — | ✓ | Non-Gaussian, so it needs the Fock backend and cannot be compiled for the machine |
| `std.two_mode_squeezing` | ✓ | — | — | ✓ | ✓ | Not part of the machine's gate set, so it runs on the simulator only |
| `std.loss_channel` | ✓ | — | — | ✓ | ✓ | A channel rather than a unitary, so it runs on the simulator only |
| `std.thermal_loss_channel` | ✓ | — | — | ✓ | ✓ | A channel rather than a unitary, so it runs on the simulator only |
| `std.measure_fock` | ✓ | — | — | — | ✓ | A photon-number measurement, so it needs the Fock backend and has no machine equivalent |
| `std.partial_trace` | ✓ | — | — | ✓ | ✓ | A partial trace, which discards a mode and so runs on the simulator only |

(sec:support-states)=

## States

| State | Held as | Simulator | Machine | Note |
| --- | --- | --- | --- | --- |
| {py:class}`~claveles.circuit.state.GaussianState` | bosonic | ✓ | — |  |
| {py:class}`~claveles.circuit.state.BosonicState` | bosonic | ✓ | — |  |
| {py:class}`~claveles.circuit.state.FockState` | fock | ✓ | — |  |
| {py:class}`~claveles.circuit.state.HardwareConstrainedSqueezedState` | — | — | ✓ | A state whose squeezing level the machine supplies, which the simulator does not have |

A mode left with no state placed on it is supplied by the machine as its own squeezed state, and is
refused by the simulator, which has no squeezing level to invent.

A circuit may mix the two representations across its modes. Which one a run ends in is settled by
{py:class}`~claveles.client.SimulatorClient`'s
{py:attr}`~claveles.client.simulator_client.SimulatorClient.state_representation`: `auto` (the default)
runs a circuit in whichever representation it needs, converting into the Fock basis before the first
operation where the circuit requires it anywhere; `bosonic` refuses such a circuit instead of
converting; and `fock` converts every bosonic state up front even where `auto` would have stayed
bosonic. A mode is converted at the cutoff its state was placed with, or at a dimension the simulator
picks where none was chosen, and the conversion truncates the state's infinite Fock tail, so the
weight lost that way is left in the state rather than normalized away.

A {py:class}`~claveles.circuit.state.FockState` placed as one component may span at most 3 modes when pure and 2 when mixed, because purifying a mixed state spends one of the available sites on the ancilla that carries the mixture.

(sec:support-backends)=

## Backends

| State representation | `single_cpu` | `multi_cpu` | `multi_cpu_peak` | `multi_cpu_shot` | `single_gpu` | `multi_gpu` |
| --- | --- | --- | --- | --- | --- | --- |
| bosonic | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fock | ✓ | single-threaded | — | — | — | — |

`single-threaded` marks a combination that runs, but not as asked: the auto-selecting CPU backend
runs a Fock circuit on one thread, because ITensor is not safe to call concurrently. Asking for a
multi-threaded CPU backend by name is refused instead, so that no one is left expecting parallelism
that cannot be delivered.

The row that applies is the representation a run *ends* in, not the one it starts in: a backend is
chosen once, so a circuit that starts bosonic and reaches the Fock basis partway through is bound
by the `fock` row for the whole of it, and cannot use a GPU backend.

Every backend in this table is reachable through {py:class}`~claveles.client.SimulatorClient`, whose
{py:attr}`~claveles.client.simulator_client.SimulatorClient.backend` option also takes `cpu` as an alias
for `multi_cpu`. The peak- and shot-level CPU backends are distinct from `multi_cpu`, which chooses
between them.

(sec:support-features)=

## Options

An option that is not an operation in its own right can be restricted in the same way. These are the
ones that are, with the representations they may be used in:

| Option | Representations |
| --- | --- |
| `post_select` | circuit |

{py:attr}`~claveles.circuit.Operation.post_select` is the parameter that fixes a measurement's outcome
instead of sampling one: a quadrature value on
{py:class}`std.MeasureHomodyne <claveles.circuit.ops.std.MeasureHomodyne>`, a photon number on
{py:class}`std.MeasureFock <claveles.circuit.ops.std.MeasureFock>`. A circuit that asks for it cannot be
converted to a graph, which has no field to carry it.
