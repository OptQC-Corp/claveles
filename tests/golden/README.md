# Golden values

Simulation results of a fixed set of reference circuits, recorded so that a later change can be shown
not to have moved them.

The upcoming state and circuit redesign removes the current API outright, which also removes the
ability to demonstrate afterwards that the numerics survived. These files are that demonstration:
they are captured while the current implementation is still in place, and every subsequent step is
checked against them.

## Layout

| Path | Contents |
|---|---|
| `cases.py` | The reference circuits and the simulator settings each is run under |
| `store.py` | Running a case, and reading or writing its record |
| `capture.py` | Command line entry point for capturing and for re-checking |
| `test_golden.py` | The comparison test |
| `data/<case>.json` | Settings captured under, plus the structure of each saved state |
| `data/<case>.npz` | The numbers |

One `.json` / `.npz` pair per case. The settings live in the JSON so they stay readable in a diff; the
arrays live in the NPZ so they stay exact.

## Cases

| Case | Watches |
|---|---|
| `bosonic_single_mode` | Homodyne sampling from a squeezed state |
| `bosonic_multi_mode_ff` | Two-mode entangling gates and a feedforward into a displacement |
| `fock_pure_gkp_kerr` | Pure multi-mode Fock state carrying a non-Gaussian gate |
| `fock_mixed_loss` | A loss channel turning the state mixed |
| `fock_mixed_partial_trace` | The other route to a mixed state, with a gate applied afterwards |
| `fock_truncated` | MPS truncation, with the discarded weight tracked |
| `post_select_homodyne` | Post-selected homodyne outcome and the state it projects onto |
| `post_select_fock` | Post-selected Fock outcome under two-photon interference |
| `bosonic_epr` | A two-mode state placed as one state, exercising the explicit mode indices on the wire |
| `fock_joint_pure` | A pure Fock state spanning two modes, through the joint-state path |
| `fock_joint_mixed` | A mixed Fock state spanning two modes, purified onto an ancilla on the way in |
| `graph_route` | The same circuit embedded into a graph representation |
| `machinery_route` | The same graph lowered to a machinery representation |

## Running the check

```shell
pytest tests/golden
```

Re-running the circuits needs the `claveles_simulator` binding. It is not installed in CI, so that part
skips there; the digest check and the well-formedness check on the stored files run everywhere.
Locally, use an interpreter whose binding was built from the C++ sources under test — rebuilding the
C++ build directory alone does not update the binding.

## Capturing

```shell
PYTHONPATH=tests python -m golden.capture --check          # re-run and report differences
PYTHONPATH=tests python -m golden.capture                  # overwrite every case
PYTHONPATH=tests python -m golden.capture --case fock_mixed_loss
```

Run `--check` first. On an unchanged implementation it reports every case as `same`, which is what
makes a fresh capture worth trusting.

## What the comparison assumes

Everything crosses the protobuf boundary as 32-bit floats — measured values, Gaussian means and
covariances, Fock amplitudes and density matrix entries alike. A captured value is therefore already
rounded to float32 before it is stored as float64, so an unchanged implementation reproduces it bit
for bit and the comparison is exact rather than tolerance-based. Reproducibility was verified by
capturing twice, and the multi-threaded backend was checked to agree with `single_cpu`.

## When a case legitimately changes

The seed is fixed and the simulator derives each shot's stream from the seed and the shot index, so a
case that gains or loses an operation does not disturb the other shots. Within a shot, though, a
change to how many random draws are consumed does move the sampled values. Cases whose values come
from sampling are marked `"sampling": true` in their JSON; for those, a mismatch may be such a change
showing through rather than a defect, and the reasoning belongs in the review. The cases marked
`"sampling": false` are post-selected or measurement-free and have no such excuse — those must keep
matching, and a mismatch there is a defect until proven otherwise.

Two planned changes are expected to move these values, and the goldens should be re-captured only once
the new behaviour has been confirmed correct by other means:

- Measurement always destroying its mode, which removes the vacuum currently reassigned to a measured
  mode, so `bosonic_single_mode` and `bosonic_multi_mode_ff` will report different saved states.
- Any change to the order in which random draws are consumed inside a shot.
