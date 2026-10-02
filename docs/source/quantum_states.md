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

(sec:quantum-states-in-the-sdk)=

# Working with Quantum States

This page is about working with the state objects: building one, reading what it holds, placing it on a
circuit and reading what a simulation gives back.

{ref}`States <sec:state-primitives>` is the catalogue that goes with it — every factory with the state it
builds, the four representations a state can be held in, and which factory can be held which way. For the
physics of the states themselves, see {ref}`Introduction to Quantum States <sec:states>`.

**{py:class}`~claveles.circuit.state.QuantumState` is the layer to work in.** Every state on this page is
built and read through it. It is a **container**: one or more separable pieces, each together with the
modes it covers. The pieces are the *basic states*, and you do not have to construct one yourself, but the
class names are worth recognizing, because you will see them in two places: a simulation result reports
which kind each component came back as, and an error message says which kind it is complaining about, like
`The component on mode(s) [0] is a BosonicState with 4 peaks`.

(sec:quantum-states-making-one)=

## Making a state

{py:class}`~claveles.circuit.state.QuantumState` has a factory for each of the usual states, and most of
them take `repr` to say how the state should be held:

```{code-cell} python
from claveles.circuit.state import QuantumState

vacuum = QuantumState.vacuum()  # bosonic, the default when repr is omitted
squeezed = QuantumState.squeezed(r=0.5, phi=0.0)
coherent = QuantumState.coherent(1 + 0.5j, repr="gaussian")
cat = QuantumState.cat(alpha=2.0, repr="bosonic")
gkp = QuantumState.gkp(repr="fock", cutoff_dim=40)

print(vacuum.n_modes, squeezed.n_modes, coherent.n_modes, cat.n_modes, gkp.n_modes)
```

{ref}`The factories <sec:state-primitives-factories>` lists every one of them with the state it builds, and
{ref}`sec:quantum-states-choosing-a-repr` which `repr` each one accepts. What the choice amounts to is
which basic state the container ends up holding, and hence which backend can run the circuit — see the
{ref}`Support Matrix <sec:support>`.

The basic states have the same factories, and a container is built out of exactly these. Worth seeing
once, so that the type names are familiar when a result or an error message mentions one:

```{code-cell} python
from claveles.circuit.state import BosonicState, FockState, GaussianState

print(BosonicState.squeezed(r=0.5, phi=0.0))
print(FockState.fock(n=2, cutoff_dim=6))
print(GaussianState.coherent(1 + 0.5j))
```

`cutoff_dim` is the number of Fock levels a mode is truncated at. `repr="fock"` requires it; the other two
keep it as the dimension the simulator would convert the state at, and the container's factory and the
basic state's own factory give the same answer:

```{code-cell} python
print(BosonicState.coherent(1 + 0.5j, cutoff_dim=40).cutoff_dims)
print(QuantumState.coherent(1 + 0.5j, cutoff_dim=40).cutoff_dims)
```

### The state the machine supplies

{py:meth}`~claveles.circuit.state.QuantumState.hardware_squeezed` is the one state you do not give a
squeezing level to, the machine fixing it and leaving you the angle:

```{code-cell} python
hardware = QuantumState.hardware_squeezed(phi=0.0)  # 0 is x-squeezed, pi/2 p-squeezed
print(hardware.n_modes, hardware.is_gaussian, hardware.cutoff_dims)
```

It is the one state the machine runs as placed and the one the simulator refuses;
{ref}`sec:quantum-states-hardware-squeezed` says why, and what it leaves unreadable.

### A state none of the factories names

A state held in the Fock basis can be given as raw amplitudes instead, with
{py:meth}`~claveles.circuit.state.QuantumState.from_state_vector` for a pure state and
{py:meth}`~claveles.circuit.state.QuantumState.from_density_matrix` for a mixed one. This is the way in for a
state the factories do not cover:

```{code-cell} python
import numpy as np

# (|0> + |3>) / sqrt(2), truncated at 6 levels.
amplitudes = np.zeros(6, dtype=np.complex128)
amplitudes[0] = amplitudes[3] = 1 / np.sqrt(2)
pure = QuantumState.from_state_vector(amplitudes, cutoff_dim=6)

# An equal mixture of |0> and |1>.
mixed = QuantumState.from_density_matrix(np.diag([0.5, 0.5, 0.0]).astype(np.complex128), cutoff_dim=3)

print(pure.is_pure, mixed.is_pure)
```

Giving several cutoffs, one per mode, makes a state spanning that many modes:

```{code-cell} python
# (|0,0> + |1,2>) / sqrt(2), with a different cutoff on each of the two modes.
joint = np.zeros((2, 3), dtype=np.complex128)
joint[0, 0] = joint[1, 2] = 1 / np.sqrt(2)
entangled = QuantumState.from_state_vector(joint, cutoff_dim=[2, 3])

print(entangled.n_modes, entangled.cutoff_dims, entangled.ket().shape)
```

Such a state is held as one component, so it is a *correlated* input: it goes onto all of its modes at
once rather than one mode at a time, and a circuit that asks for fewer modes than it spans is refused.
{ref}`sec:quantum-states-placing` shows the placement itself.

{py:meth}`~claveles.circuit.state.QuantumState.ket` and {py:meth}`~claveles.circuit.state.QuantumState.dm` go
the other way, turning a state into an array, and these two factories turn an array back into a state.
So one undoes the other: `QuantumState.from_state_vector(state.ket(), state.cutoff_dims)` gives back a
state with the same amplitudes, and `from_density_matrix` does the same for
{py:meth}`~claveles.circuit.state.QuantumState.dm`. What comes back is always a single component numbered
from mode 0, so a state that had been several components arrives joined into one — the same state, held
in one piece instead of several.

Both take the array as given rather than rescaling it, which is the same convention the rest of the
Fock representation follows: a squared norm (or a trace) below 1 is accepted as a truncated state,
and one above 1 is rejected as an unnormalized input by the simulator when the circuit runs, not when
the state is built. Pass `normalize=True` to rescale to exactly 1 instead —
over the whole array for a joint state, and by the trace of the whole matrix for a density matrix,
rather than mode by mode.

## Reading a container

A container reports the modes it covers, and lets you take one mode out:

```{code-cell} python
state = QuantumState.epr(r=1.0) & QuantumState.vacuum()
print("modes      :", state.modes)
print("n_modes    :", state.n_modes)
print("components :", [(modes, type(basic).__name__) for basic, modes in state.states])

mode2 = state.extract_mode(2)
print("mode 2 only:", mode2.modes, type(mode2.state_list[0]).__name__)
```

{py:attr}`~claveles.circuit.state.QuantumState.states` pairs each basic state with the modes it covers, and
{py:attr}`~claveles.circuit.state.QuantumState.state_list` is just the basic states. A component spanning two
modes appears once, covering both, which is how the correlation between them is kept:
{py:meth}`~claveles.circuit.state.QuantumState.epr` above is one component over two modes, so
{py:attr}`~claveles.circuit.state.QuantumState.states` has two entries rather than three.

{py:meth}`~claveles.circuit.state.QuantumState.extract_mode` traces out everything else, so it answers
"what does this one mode look like" rather than "give me the piece that mode belongs to".

A container also says what it knows about the state as a whole:

```{code-cell} python
print("squeezed  :", squeezed.is_pure, squeezed.is_gaussian)
print("cat       :", cat.is_pure, cat.is_gaussian)
print("fock GKP  :", gkp.is_pure, gkp.is_gaussian)
```

{py:attr}`~claveles.circuit.state.QuantumState.is_pure` and
{py:attr}`~claveles.circuit.state.QuantumState.is_gaussian` are True only when every component says so, so
they mean "certainly pure" and "certainly Gaussian" rather than "not known to be otherwise". A cat
state is reported as neither, even though it is in fact a pure state: it is held as four Gaussian
peaks, and deciding purity from those means looking at how they interfere, which is not attempted.

(sec:quantum-states-mean-and-cov)=

### The mean and the covariance

A state that is Gaussian throughout reports its phase space directly, with
{py:attr}`~claveles.circuit.state.QuantumState.mean` and {py:attr}`~claveles.circuit.state.QuantumState.cov`:

```{code-cell} python
state = QuantumState.epr(r=0.5) & QuantumState.coherent(1.0)
print("modes :", state.modes)
print("mean  :", state.mean.real.round(6))
print("cov   :", state.cov.shape)
```

Both are ordered as every mode's $x$ in ascending mode order followed by every mode's $p$ — the order
{py:meth}`~claveles.circuit.state.QuantumState.gaussian` takes and
{py:meth}`~claveles.circuit.state.QuantumState.wigner_func` takes, so
`QuantumState.gaussian(state.mean, state.cov)` rebuilds the state. Note that this is *not* each
component's own vector laid end to end: a component orders its own $x$'s then its own $p$'s, and the
container gathers the $x$'s of every component ahead of all the $p$'s.

Because the components are separable, the blocks of
{py:attr}`~claveles.circuit.state.QuantumState.cov` between two different components are zero. The correlations a state does carry live inside one component, which is why
{py:meth}`~claveles.circuit.state.QuantumState.epr` above is a single component over two modes.

A state that is not Gaussian throughout has no single mean and covariance. Rather than return an
approximation, both properties raise, and the message names the offending component and the modes it sits
on. Three kinds of component cause this: a cat state, which is four peaks rather than one; a Fock state,
which is held as amplitudes instead (use {py:meth}`~claveles.circuit.state.QuantumState.ket` or
{py:meth}`~claveles.circuit.state.QuantumState.dm`); and the machine's squeezed state, whose level it does not
disclose. Watch out for that last one: it answers
{py:attr}`~claveles.circuit.state.QuantumState.is_gaussian` with True and still has no mean, so that flag does
not tell you whether these can be read.

How you asked for the state does not come into it. A mode held as a
{py:class}`~claveles.circuit.state.GaussianState` and the same mode held as a
{py:class}`~claveles.circuit.state.BosonicState` answer both properties alike:

```{code-cell} python
as_gaussian = QuantumState.coherent(1 + 0.5j, repr="gaussian")
as_bosonic = QuantumState.coherent(1 + 0.5j)

print("mean, gaussian:", as_gaussian.mean)
print("mean, bosonic :", as_bosonic.mean)
print("cov agrees    :", np.allclose(as_gaussian.cov, as_bosonic.cov))
print("both Gaussian :", as_gaussian.is_gaussian, as_bosonic.is_gaussian)
```

The two also reach the same backend, which is what
{ref}`sec:quantum-states-gaussian-and-bosonic` is about.

### The Fock expansion

When every component is a {py:class}`~claveles.circuit.state.FockState`, the container can hand out the
whole state's amplitudes with {py:meth}`~claveles.circuit.state.QuantumState.ket` and its density matrix
with {py:meth}`~claveles.circuit.state.QuantumState.dm`:

```{code-cell} python
pair = QuantumState.fock(n=1, cutoff_dim=3) & QuantumState.coherent(0.5, repr="fock", cutoff_dim=4)
print("cutoff per mode:", pair.cutoff_dims)
print("ket:", pair.ket().shape)
print("dm :", pair.dm().shape)
```

{py:attr}`~claveles.circuit.state.QuantumState.cutoff_dims` gives one cutoff per mode, in ascending mode
order, which for a state held in the Fock basis is exactly the shape of
{py:meth}`ket <claveles.circuit.state.QuantumState.ket>`. A state of one mode can use the singular
{py:attr}`~claveles.circuit.state.QuantumState.cutoff_dim` instead. A wider state raises instead of
answering, because each mode carries its own cutoff and no one of them stands for the rest.

Both describe the state **as a whole**, not one mode at a time.
{py:meth}`ket <claveles.circuit.state.QuantumState.ket>` has one axis per mode, in ascending mode order, so
its shape is just the modes' cutoffs. {py:meth}`dm <claveles.circuit.state.QuantumState.dm>` is the density
matrix of that joint state, so each of its two axes runs over the whole joint basis — all
$3 \times 4 = 12$ combinations of the two modes' levels — which is why it comes out $12 \times 12$. It is
the flattened {py:meth}`ket <claveles.circuit.state.QuantumState.ket>` against itself, with the first mode as
the most significant axis, the same convention {py:meth}`~claveles.circuit.state.FockState.state_vector` and
{py:meth}`~claveles.circuit.state.FockState.density_matrix` use for one component.

One mode on its own is a different question, and
{py:meth}`~claveles.circuit.state.QuantumState.extract_mode` is what answers it: it traces the other modes
out, leaving a state whose density matrix is that mode's own cutoff square.

```{code-cell} python
print("mode 0 alone:", pair.extract_mode(0).dm().shape)
print("mode 1 alone:", pair.extract_mode(1).dm().shape)
```

The same holds for a mode that shares its component with another, as the two modes of `entangled` above
do. The other mode is traced out of that component, and what is left is in general mixed, so it is held as
a density matrix:

```{code-cell} python
mode1 = entangled.extract_mode(1)
print("pure:", mode1.is_pure)
print(mode1.dm().real.round(3))
```

Neither {py:meth}`ket <claveles.circuit.state.QuantumState.ket>` nor
{py:meth}`dm <claveles.circuit.state.QuantumState.dm>` is normalized, and the weights of the components
multiply, so divide by the norm or the trace if you need a normalized state. A mixed component leaves the
state without a {py:meth}`ket <claveles.circuit.state.QuantumState.ket>`, and a component held in any other
representation is refused by both: converting a Gaussian or bosonic component into the Fock basis is not
done here.

### The Wigner function

{py:meth}`~claveles.circuit.state.QuantumState.draw_wigner` plots one mode, and
{py:meth}`~claveles.circuit.state.QuantumState.wigner_func` returns the function itself, of the $2n$
quadratures $x_0, \dots, x_{n-1}, p_0, \dots, p_{n-1}$ in ascending mode order:

```{code-cell} python
wigner = QuantumState.epr(r=0.5).wigner_func()
print("at the origin      :", wigner(0.0, 0.0, 0.0, 0.0))
# The two x quadratures are correlated, so equal values are the likelier pair.
print("x0 = x1 = 1        :", wigner(1.0, 1.0, 0.0, 0.0))
print("x0 = 1, x1 = -1    :", wigner(1.0, -1.0, 0.0, 0.0))

QuantumState.cat(alpha=2.0).draw_wigner(title="cat state");
```

{py:meth}`~claveles.circuit.state.QuantumState.wigner_func` broadcasts its arguments, so a whole mesh can be
passed in one call. Bosonic and Gaussian components may span any number of modes; a Fock component is
limited to one mode, so take one mode out of a wider one with
{py:meth}`~claveles.circuit.state.QuantumState.extract_mode` first. The squeezed state the machine supplies
has no Wigner function here, its squeezing level being the machine's to decide. Drawing is a 2-D plot
either way, so {py:meth}`~claveles.circuit.state.QuantumState.draw_wigner` needs a single-mode state —
{py:meth}`~claveles.circuit.state.QuantumState.extract_mode` first if the state is wider.

### Renumbering the modes

{py:meth}`~claveles.circuit.state.QuantumState.idx_shift` moves every mode by the same amount, and
{py:meth}`~claveles.circuit.state.QuantumState.idx_align` closes up the gaps a measured-away mode leaves
behind. Both return a new container, aliases and all, and leave the original alone:

```{code-cell} python
placed = QuantumState.concatenate(
    [QuantumState.squeezed(r=0.5), QuantumState.vacuum()],
    mode_idx_list=[[0], [3]],
    mode_names={3: "ancilla"},
)
print("as placed :", placed.mode_idx_list, placed.mode_names)
print("shifted   :", placed.idx_shift(2).mode_idx_list, placed.idx_shift(2).mode_names)
print("aligned   :", placed.idx_align().mode_idx_list, placed.idx_align().mode_names)
```

## Combining states

Two containers can be put side by side with `&`, which is
{py:meth}`~claveles.circuit.state.QuantumState.tensor`. The state on the left keeps the mode indices it has,
and the one on the right is shifted to start above the highest of them, so a state built on its own —
which sits on mode 0 — lands on the mode after:

```{code-cell} python
pair = QuantumState.squeezed(r=0.5) & QuantumState.vacuum()
print(pair.modes)
```

That is the same rule `&` on a {py:class}`~claveles.circuit.CircuitRepr` follows, and it matters when the
indices mean something. A state a simulation reported covers the modes the circuit used,
with a gap where a mode was measured away, and laying another state beside it leaves those modes where
they were rather than closing the gap:

```{code-cell} python
gapped = QuantumState.concatenate(
    [QuantumState.squeezed(r=0.5), QuantumState.vacuum()],
    mode_idx_list=[[0], [2]],
)

print("as placed :", gapped.modes)
print("combined  :", (gapped & QuantumState.vacuum()).modes)
print("aligned   :", (gapped & QuantumState.vacuum()).idx_align().modes)
```

{py:meth}`~claveles.circuit.state.QuantumState.idx_align` is how the gaps are closed when you do want them
closed. {py:meth}`a.tensor(b) <claveles.circuit.state.QuantumState.tensor>` and
{py:meth}`QuantumState.tensor(a, b) <claveles.circuit.state.QuantumState.tensor>` both work and mean the same
thing.

{py:meth}`~claveles.circuit.state.QuantumState.concatenate` takes a list instead, and
{py:attr}`~claveles.circuit.state.QuantumState.mode_idx_list` says which modes the result should cover —
useful when the states belong on modes you have already decided. It joins every state it is given, so it
is called on the class rather than through one of them:

```{code-cell} python
placed = QuantumState.concatenate(
    [QuantumState.squeezed(r=0.5), QuantumState.vacuum()],
    mode_idx_list=[[3], [7]],
)
print(placed.modes)
```

(sec:quantum-states-placing)=

## Placing a state on a circuit

A state is placed the same way an operation is applied, and it has to be the first thing that touches
its modes — see {ref}`sec:circuit-placing-states`. A component spanning several modes is placed on
several modes at once, which is how a correlated input is expressed:

```{code-cell} python
from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std

circuit = CircuitRepr("correlated_input")
circuit.Q(0, 1) | QuantumState.epr(r=1.0)     # one component over two modes
circuit.Q(0, 1) | std.BeamSplitter(theta=0.3, phi=0.0)

print([(modes, type(basic).__name__) for modes, basic in circuit.placed_components])
```

A {py:class}`~claveles.circuit.state.FockState` placed as one component may span at most three modes when
pure and two when mixed, because purifying a mixed state spends one of the available sites on the
ancilla that carries the mixture.

(sec:quantum-states-from-a-simulation)=

## What comes back from a simulation

A simulation reports one {py:class}`~claveles.circuit.state.QuantumState` per saved shot, holding the modes
that were not measured. Reading it is the same as reading any other container:

```{code-cell} python
from claveles.client import SimulatorClient

circuit = CircuitRepr("readout")
circuit.Q(0) | QuantumState.squeezed(r=0.5)
circuit.Q(1) | QuantumState.vacuum()
circuit.Q(0) | std.MeasureHomodyne(0.0)

result = SimulatorClient(n_shots=1, state_save_policy="all").run(circuit)
saved = result.shot_results[0].state
print("surviving modes:", saved.modes)
print("held as        :", type(saved.state_list[0]).__name__)
```

Mode 0 was measured, so it is gone and mode 1 is what is left. Note that the surviving mode keeps its
original index: the container is indexed by the modes the circuit used, not renumbered from zero.

Every component comes back as a {py:class}`~claveles.circuit.state.BosonicState` or a
{py:class}`~claveles.circuit.state.FockState`, whichever representation the run ended in. A mode placed as a
{py:class}`~claveles.circuit.state.GaussianState` comes back as the single-peak
{py:class}`~claveles.circuit.state.BosonicState` it was serialized as, so `repr="gaussian"` on the way in
does not make it the way out.

See {ref}`sec:simulation-reading-results` for the rest of what a shot reports.
