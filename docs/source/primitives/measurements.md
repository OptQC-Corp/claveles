(sec:measurements)=

# Measurements

In this page, we enumerate the measurement operations of both the circuit and graph representations. A
measurement is how a result leaves a circuit: it consumes the mode it reads, and the value it returns is
what a shot reports.

{ref}`Introduction to Measurement <sec:measurement>` introduces the two measurements themselves — what
each one reads, how its outcome is distributed, and why the two are complementary — starting from what a
quadrature is and needing nothing from the SDK. Read it first if homodyne and Fock-number measurement are
new to you; this page gives the operations as the SDK offers them.

The conventions of the {ref}`Gates & Operations <sec:gates>` page hold here as well: {math}`\hbar = 1`,
{math}`\hat{a}=(\hat{x}+i\hat{p})/\sqrt{2}`, and a symbol takes an {math}`\text{intr}` subscript only
where the `intrinsic` operation is a different function of its arguments than the `std` one — see
{ref}`the conventions <sec:gates-conventions>` and {ref}`the symbol conventions <sec:gates-symbols>` there.
Each operation carries a {{ bdg_hw }} badge when
it can also run on the real machine, followed by a {{ bdg_bosonic }} or {{ bdg_fock }} badge for every
state representation it can be applied to; hover over a badge to see what it means. The
{ref}`Support Matrix <sec:support>` lists the same answers for every operation in one table.

(sec:measurements-std)=

## Standard measurements (circuit representation)

### {py:class}`std.MeasureHomodyne <claveles.circuit.ops.std.MeasureHomodyne>`

{{ bdg_hw }} {{ bdg_bosonic }} {{ bdg_fock }}

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\phi`
*   - Operation
    - Measure {math}`\hat{x} \cos \phi + \hat{p} \sin \phi`
```

Implemented using the measurement operation {math}`\hat{M}_\text{intr}` defined in {ref}`sec:gate-measurement-symbol` with {math}`\theta = \frac{\pi}{2} - \phi`.

### {py:class}`std.MeasureFock <claveles.circuit.ops.std.MeasureFock>`

{{ bdg_fock }}

```{list-table}
:stub-columns: 1

*   - Parameters
    -
*   - Operation
    - Measure {math}`\hat{n}`
*   - Note
    - Circuit representation (simulator) only, for Fock-represented states. Not available in graph or machinery representations.
```

The measurement projects the mode onto a Fock state and returns that state's photon number.

Measuring a mode consumes it: the mode is discarded and cannot be operated on again. Set `post_select=n` to fix the outcome to a given number of photons $n$ instead of sampling.

(sec:measurements-intrinsic)=

## Intrinsic measurement

The operations in this section (`intrinsic.XXX`) are used as low-level program elements that can be implemented natively on quad-rail lattice (QRL) CV-QC hardware. When building CV quantum circuits, users can typically rely on the {ref}`standard operations <sec:measurements-std>` above instead.

(sec:gate-measurement-symbol)=

### {py:class}`intrinsic.Measurement <claveles.circuit.ops.intrinsic.Measurement>`

{{ bdg_hw }} {{ bdg_bosonic }} {{ bdg_fock }}

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`
*   - Operation
    - {math}`\hat{M}_\text{intr}\left(\theta\right)`: Measure {math}`\hat{x} \sin \theta+\hat{p} \cos \theta`
```

```{attention}
When converted to the machinery representation, the setting is that feedforward is not performed immediately after this operation. If feedforward is required, either {ref}`graph.ops.Initialization <sec:gate-graph-repr-initialization>` in graph representation should be used instead, or the feedforward matrices {py:attr}`ff_coeff_matrix_k_plus_1 <claveles.machinery.program.MachineryRepr.ff_coeff_matrix_k_plus_1>` and {py:attr}`ff_coeff_matrix_k_plus_n <claveles.machinery.program.MachineryRepr.ff_coeff_matrix_k_plus_n>` in the machinery representation must be manually set after the conversion.
```

(sec:measurements-in-graph-repr)=

## Graph-representation measurements

The graph representation has a single gate set, and its parametrization is the intrinsic one, so the
symbols below are written bare: {math}`\hat{M}` here is the measurement of {math}`\theta`, which in
circuit-representation terms is {math}`\hat{M}_\text{intr}`. See
{ref}`the symbol conventions <sec:gates-symbols>`.

The measurement angles are shown only for the through case.
They are omitted for the swap case, as swapping the input modes within each macronode is achieved by exchanging the first two measurement angles, as described in {ref}`sec:gates-graph-through-and-swap`.

### {py:class}`graph.ops.Measurement <claveles.graph.ops.Measurement>`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`
*   - Operation
    - {math}`M\left(\theta\right)`: Measure {math}`\hat{x} \sin \theta+\hat{p} \cos \theta`
*   - Measurement angles (through)
    - {math}`(\theta, \theta, \theta, \theta)`
```

The measurement operation {math}`M` is defined in {ref}`sec:gate-measurement-symbol`.

```{attention}
When converted to the machinery representation, the setting is that feedforward is not performed immediately after this operation. If feedforward is required, either {ref}`graph.ops.Initialization <sec:gate-graph-repr-initialization>` should be used instead, or the feedforward matrices {py:attr}`ff_coeff_matrix_k_plus_1 <claveles.machinery.program.MachineryRepr.ff_coeff_matrix_k_plus_1>` and {py:attr}`ff_coeff_matrix_k_plus_n <claveles.machinery.program.MachineryRepr.ff_coeff_matrix_k_plus_n>` in the machinery representation must be manually set after the conversion.
```

(sec:gate-graph-repr-initialization)=

### {py:class}`graph.ops.Initialization <claveles.graph.ops.Initialization>`

```{list-table}
:stub-columns: 1

*   - Parameters
    - {math}`\theta`
*   - Operation
    - {math}`M\left(\theta\right)`: Measure {math}`\hat{x} \sin \theta+\hat{p} \cos \theta` and initialize a mode with a {ref}`squeezing angle <sec:squeezing-level-and-squeezing-angle>` {math}`\theta + \frac{\pi}{2}`
*   - Measurement angles (through)
    - {math}`(\theta, \theta, \theta, \theta)`
```

The measurement operation {math}`M` is defined in {ref}`sec:gate-measurement-symbol`. This is the
operation to place where the measurement is followed immediately by feedforward, which
{ref}`graph.ops.Measurement <sec:measurements-in-graph-repr>` above does not arrange for.

A macronode whose four measurement angles are equal (modulo {math}`2\pi`) does two things at once. It
measures the modes coming into it, and the modes it passes on no longer carry any information about
them: they are fresh squeezed states. A measurement macronode is therefore an initialization macronode
as well, and every mode in the graph and machinery representations starts from one. What the macronode
can choose is the squeezing angle of the modes it sends on, {math}`\theta + \frac{\pi}{2}`; their
squeezing level is set by the machine or by the simulator's options, never by the macronode.

The measurement angle and the squeezing angle are tied together, so a single macronode rarely performs
both the measurement and the initialization you want. Programs usually keep the two roles apart, one
macronode for each.
