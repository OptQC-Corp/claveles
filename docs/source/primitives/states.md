(sec:state-primitives)=

# States

In this page, we enumerate the states {py:class}`~claveles.circuit.state.QuantumState` can build, and the
representations one of them can be held in.

{ref}`Introduction to Quantum States <sec:states>` introduces the states themselves — what a squeezed
state *is*, and what its Wigner function looks like — starting from what a mode is and needing nothing
from the SDK. Read it first if continuous-variable states are new to you; the table below links each
factory to the state it builds there. For building these objects, reading them and placing them on a
circuit, see {ref}`Working with Quantum States <sec:quantum-states-in-the-sdk>`.

A {py:class}`~claveles.circuit.state.QuantumState` is a **container**: one or more separable pieces, each
together with the modes it covers. The pieces are the **basic states**, of which there are four kinds, and
every factory below builds a container holding one of them.

(sec:state-primitives-factories)=

## The factories

Each factory builds the state its name says, on as many modes as the fourth column gives. The state itself
is introduced on the page the second column links to:

| Factory | Builds | Parameters | Modes |
| --- | --- | --- | --- |
| {py:meth}`~claveles.circuit.state.QuantumState.vacuum` | the {ref}`vacuum state <sec:states-vacuum>` | — | 1 |
| {py:meth}`~claveles.circuit.state.QuantumState.coherent` | a {ref}`coherent state <sec:states-coherent>` of amplitude {math}`\alpha` | `alpha` | 1 |
| {py:meth}`~claveles.circuit.state.QuantumState.squeezed` | a {ref}`squeezed vacuum state <sec:states-squeezed>` | `r`, `phi` | 1 |
| {py:meth}`~claveles.circuit.state.QuantumState.fock` | a {ref}`number state <sec:states-fock>` | `n`, and `r` for the quality of the bosonic approximation | 1 |
| {py:meth}`~claveles.circuit.state.QuantumState.cat` | a {ref}`cat state <sec:states-cat>` | `alpha`, `parity` | 1 |
| {py:meth}`~claveles.circuit.state.QuantumState.gkp` | a finite-energy approximate {ref}`GKP state <sec:states-gkp>` | `theta`, `phi`, `epsilon`, `ampl_cutoff` | 1 |
| {py:meth}`~claveles.circuit.state.QuantumState.epr` | an {ref}`EPR state <sec:states-epr>`, the two-mode squeezed vacuum | `r`, `phi` | 2 |
| {py:meth}`~claveles.circuit.state.QuantumState.gaussian` | any {ref}`Gaussian state <sec:states-gaussian>`, from its phase space | `mean`, `cov` | half the length of `mean` |
| {py:meth}`~claveles.circuit.state.QuantumState.from_state_vector` | any pure state, from its Fock amplitudes | `state_vector`, `normalize` | one per entry of `cutoff_dim` |
| {py:meth}`~claveles.circuit.state.QuantumState.from_density_matrix` | any mixed state, from its Fock-basis density matrix | `density_matrix`, `normalize` | one per entry of `cutoff_dim` |
| {py:meth}`~claveles.circuit.state.QuantumState.hardware_squeezed` | the squeezed state the machine supplies | `phi` | 1 |

One mean and one covariance describe a Gaussian state of *any* number of modes, which is why
{py:meth}`~claveles.circuit.state.QuantumState.gaussian` is the one factory whose width is not fixed:
`mean` is a single vector holding every mode's {math}`x`-mean followed by every mode's {math}`p`-mean, so
for {math}`n` modes it is {math}`2n` entries long, and `cov` is the {math}`2n \times 2n` matrix over those
same quadratures. A `mean` of two entries with a {math}`2\times2` `cov` is one mode; four entries with a
{math}`4\times4` `cov` is two modes, and the off-diagonal blocks are then the correlation between them.
{ref}`sec:quantum-states-mean-and-cov` gives the ordering in full.

The {ref}`thermal equilibrium state <sec:states-thermal>` has no factory of its own: build it with
{py:meth}`~claveles.circuit.state.QuantumState.gaussian`, whose covariance
{math}`\frac{\hbar}{2}(2\bar{n}+1)\mathbf{I}` is what
{py:class}`std.ThermalLossChannel <claveles.circuit.ops.std.ThermalLossChannel>` relaxes a mode to.

Every factory but the last three also takes `repr` and `cutoff_dim`, which say how the state is to be
held. The basic states have factories of the same names for the states each of them can hold, so most of
the table can be built one layer down as well; the exceptions are that only
{py:class}`~claveles.circuit.state.FockState` has `epr`, and that a Gaussian state of your own is built with
the {py:class}`~claveles.circuit.state.GaussianState` constructor, which takes the same `mean` and `cov`.

(sec:quantum-states-representations)=

## How a state is held

A **basic state** is one separable piece of a state, and does not know which modes it sits on. There are
four kinds, and `repr` chooses between the first three:

* {py:class}`~claveles.circuit.state.BosonicState` (`repr="bosonic"`) holds the state as a weighted sum of
  Gaussian peaks. This is the default nearly everywhere, and it is what a cat or a GKP state needs: a
  single Gaussian cannot describe either, a sum of several can.
* {py:class}`~claveles.circuit.state.GaussianState` (`repr="gaussian"`) holds one mean vector and one
  covariance matrix, and so says up front that the state is Gaussian.
* {py:class}`~claveles.circuit.state.FockState` (`repr="fock"`) holds the state in the Fock basis,
  truncated at `cutoff_dim` levels. This is the one that can hold a state the Gaussian description cannot
  reach exactly, and the one the non-Gaussian gates need.
* {py:class}`~claveles.circuit.state.HardwareConstrainedSqueezedState` names the squeezed state the machine
  supplies rather than carrying one, and has no `repr` to choose;
  {ref}`sec:quantum-states-hardware-squeezed` below is about it.

Which kind a container holds is what a simulation result reports for each of its components, and what an
error message names when it complains about one, as in
`The component on mode(s) [0] is a BosonicState with 4 peaks`.

(sec:quantum-states-choosing-a-repr)=

## Which `repr` a factory takes

Every factory takes `"bosonic"`, and it is the default for all of them but
{py:meth}`~claveles.circuit.state.QuantumState.fock`, which defaults to `"fock"` — a number state being
exact there and only approximate as a Gaussian mixture. `"gaussian"` and `"fock"` are available only
where the state can be written that way, so asking for one a state does not fit raises `ValueError`
rather than returning an approximation. **default** below marks what each factory holds the state as
when `repr` is left out:

| Factory | `"bosonic"` | `"gaussian"` | `"fock"` |
| --- | --- | --- | --- |
| {py:meth}`~claveles.circuit.state.QuantumState.vacuum` | ✓ **default** | ✓ | ✓ |
| {py:meth}`~claveles.circuit.state.QuantumState.coherent` | ✓ **default** | ✓ | ✓ |
| {py:meth}`~claveles.circuit.state.QuantumState.squeezed` | ✓ **default** | ✓ | ✓ |
| {py:meth}`~claveles.circuit.state.QuantumState.epr` | ✓ **default** | ✓ | ✓ |
| {py:meth}`~claveles.circuit.state.QuantumState.gaussian` | ✓ **default** | ✓ | — |
| {py:meth}`~claveles.circuit.state.QuantumState.cat` | ✓ **default** | — | ✓ |
| {py:meth}`~claveles.circuit.state.QuantumState.gkp` | ✓ **default** | — | ✓ |
| {py:meth}`~claveles.circuit.state.QuantumState.fock` | ✓ | — | ✓ **default** |

`"gaussian"` is refused by {py:meth}`~claveles.circuit.state.QuantumState.cat`,
{py:meth}`~claveles.circuit.state.QuantumState.gkp` and {py:meth}`~claveles.circuit.state.QuantumState.fock`
because a cat, a GKP and a number state are not Gaussian: a single mean and covariance cannot describe
them, whereas a sum of several can, which is what the bosonic representation is.

`"fock"` is refused by {py:meth}`~claveles.circuit.state.QuantumState.gaussian` alone, and there the reason
is the mean and covariance you hand it: converting those into the Fock basis is not implemented on this
side. Every other factory builds its state in whichever basis is asked for.

Three factories are missing from the table, taking no `repr` at all:
{py:meth}`~claveles.circuit.state.QuantumState.from_state_vector` and
{py:meth}`~claveles.circuit.state.QuantumState.from_density_matrix`, because an array of Fock amplitudes is
only expressible in the Fock basis, and
{py:meth}`~claveles.circuit.state.QuantumState.hardware_squeezed`, which names a squeezing level rather than
carrying a state.

`repr="fock"` also takes `cutoff_dim`, the number of Fock levels the mode is truncated at, and requires
it. The other two keep it as the *conversion* cutoff — the dimension the simulator converts the state at
if the run ends up in the Fock representation. Leaving it unset there lets the simulator pick a dimension
at conversion time, which is what {py:class}`~claveles.warnings.StateConversionWarning` reports, and which
{py:attr}`~claveles.circuit.state.QuantumState.cutoff_dims` reports as `None` for that mode. Where a state
spans several modes — {py:meth}`~claveles.circuit.state.QuantumState.epr` always,
{py:meth}`~claveles.circuit.state.QuantumState.gaussian` as wide as its `mean` — `cutoff_dim` is the
dimension of *each* of those modes.

(sec:quantum-states-gaussian-and-bosonic)=

## `"gaussian"` and `"bosonic"` reach the same backend

The simulator has two backends, bosonic and Fock, and a {py:class}`~claveles.circuit.state.GaussianState`
runs on the bosonic one: placing it on a circuit serializes it as the single-peak
{py:class}`~claveles.circuit.state.BosonicState` that carries it exactly. So `repr="gaussian"` chooses what
*you* hold, not how the circuit runs, and it is not a third value of
{py:class}`~claveles.client.SimulatorClient`'s
{py:attr}`~claveles.client.simulator_client.SimulatorClient.state_representation`, which is `"auto"`,
`"bosonic"` or `"fock"`.

Reading the state does not depend on the choice either: `mean` and `cov` are read from the container in
the same way whichever of the two it holds, so `repr` decides how the state is stored and nothing about
how you read it.

Holding it as a {py:class}`~claveles.circuit.state.GaussianState` does buy one thing: it says the state is
Gaussian up front. That constructor takes a mean and a covariance and nothing else, so a non-Gaussian
state cannot end up in one by accident.

A simulation reports a {py:class}`~claveles.circuit.state.BosonicState` for such a mode either way, never a
{py:class}`~claveles.circuit.state.GaussianState`; see {ref}`sec:quantum-states-from-a-simulation`. A
single-peak {py:class}`~claveles.circuit.state.BosonicState` answers
{py:attr}`~claveles.circuit.state.BosonicState.is_gaussian` with True, so nothing about the state is lost,
and {py:meth}`get_gaussian_state(0) <claveles.circuit.state.BosonicState.get_gaussian_state>` hands the peak
back.

(sec:quantum-states-hardware-squeezed)=

## The state the machine supplies

One state is not yours to specify. On real hardware the squeezing level is fixed by the machine and only
the angle is a choice, which is what {py:meth}`~claveles.circuit.state.QuantumState.hardware_squeezed`
stands for: `phi=0` is an x-squeezed state and {math}`\phi = \pi/2` a p-squeezed one.

It takes no `repr` and no `cutoff_dim`: it carries neither amplitudes nor a covariance, so there is
nothing to hold another way and nothing to truncate. For the same reason it has no
{py:attr}`~claveles.circuit.state.QuantumState.mean` and no
{py:attr}`~claveles.circuit.state.QuantumState.cov`, while still answering
{py:attr}`~claveles.circuit.state.QuantumState.is_gaussian` with True.

It is the one state {py:class}`~claveles.client.OptQCClient` runs as placed — a circuit carrying any other
is refused on conversion (see {ref}`sec:graph-initial-states`) — and the one state
{py:class}`~claveles.client.SimulatorClient` refuses, having no squeezing level to assume for it. It is also
what {py:meth}`~claveles.circuit.CircuitRepr.proto`'s `default_state` normally fills an open mode with.
