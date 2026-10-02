#!/usr/bin/env python3
# Copyright 2026 OptQC Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the support-matrix documentation page from :mod:`claveles.support`.

The tables on the page are not maintained by hand: they are printed from the same table the SDK
consults when it accepts or refuses a circuit, so the documentation cannot claim something the code
does not do.

Usage:
  python tools/gen_support_page.py           # write docs/source/support.md
  python tools/gen_support_page.py --check   # exit 1 if the file is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from claveles.support import (
    BACKEND_SUPPORT,
    FEATURE_SUPPORT,
    FOCK_JOINT_STATE_MODE_LIMIT,
    OPERATION_SUPPORT,
    STATE_SUPPORT,
    Backend,
    Representation,
    StateRepresentation,
)

PAGE = Path(__file__).parent.parent / "docs" / "source" / "support.md"

_REPRESENTATIONS: tuple[Representation, ...] = ("circuit", "graph", "machinery")
_STATE_REPRESENTATIONS: tuple[StateRepresentation, ...] = ("bosonic", "fock")
_BACKENDS: tuple[Backend, ...] = (
    "single_cpu",
    "multi_cpu",
    "multi_cpu_peak",
    "multi_cpu_shot",
    "single_gpu",
    "multi_gpu",
)

_YES = "✓"
_NO = "—"
_VERDICT = {"supported": _YES, "rejected": _NO, "downgraded": "single-threaded"}


def _sentence(reason: str) -> str:
    """Capitalize a reason phrase without touching the rest of it.

    Args:
        reason: A phrase that reads as the middle of a sentence, possibly empty.

    Returns:
        str: The phrase with its first letter capitalized, so that "non-Gaussian" keeps its capital G.
    """
    return reason[0].upper() + reason[1:] if reason else ""


def _operation_rows() -> list[str]:
    """Return one table row per operation.

    Returns:
        list[str]: Markdown table rows, in the order the table lists them.
    """
    rows = []
    for op_name, support in OPERATION_SUPPORT.items():
        cells = [_YES if name in support.representations else _NO for name in _REPRESENTATIONS]
        cells += [_YES if name in support.state_representations else _NO for name in _STATE_REPRESENTATIONS]
        note = _sentence(support.reason)
        rows.append(f"| `{op_name}` | {' | '.join(cells)} | {note} |")
    return rows


def _state_rows() -> list[str]:
    """Return one table row per basic state.

    Returns:
        list[str]: Markdown table rows.
    """
    rows = []
    for name, support in STATE_SUPPORT.items():
        held_as = support.state_representation or _NO
        rows.append(
            f"| {{py:class}}`~claveles.circuit.state.{name}` | {held_as} | "
            f"{_YES if support.simulator else _NO} | {_YES if support.hardware else _NO} | "
            f"{_sentence(support.reason)} |"
        )
    return rows


def _backend_rows() -> list[str]:
    """Return one table row per state representation, with a column per backend.

    Returns:
        list[str]: Markdown table rows.
    """
    rows = []
    for state_representation in _STATE_REPRESENTATIONS:
        verdicts = BACKEND_SUPPORT[state_representation]
        cells = [_VERDICT[verdicts[backend]] for backend in _BACKENDS]
        rows.append(f"| {state_representation} | {' | '.join(cells)} |")
    return rows


def render() -> str:
    """Return the whole page.

    Returns:
        str: Markdown source for ``docs/source/support.md``.
    """
    op_header = " | ".join(_REPRESENTATIONS) + " | " + " | ".join(_STATE_REPRESENTATIONS)
    op_rule = " | ".join(["---"] * (len(_REPRESENTATIONS) + len(_STATE_REPRESENTATIONS)))
    backend_header = " | ".join(f"`{backend}`" for backend in _BACKENDS)
    backend_rule = " | ".join(["---"] * len(_BACKENDS))
    fock_limit = (
        "A {py:class}`~claveles.circuit.state.FockState` placed as one component may span at most "
        f"{FOCK_JOINT_STATE_MODE_LIMIT[True]} modes when pure and {FOCK_JOINT_STATE_MODE_LIMIT[False]} when mixed, "
        "because purifying a mixed state spends one of the available sites on the ancilla that carries the mixture."
    )
    feature_rows = "\n".join(
        f"| `{feature}` | {', '.join(sorted(representations))} |" for feature, representations in FEATURE_SUPPORT.items()
    )

    return f"""(sec:support)=

# Support Matrix

Which operations and states can be used where. This page is generated from {{py:mod}}`claveles.support`,
the table the SDK itself consults, so it says exactly what the SDK will accept or refuse.

`{_YES}` means supported and `{_NO}` means not supported.

```{{note}}
Do not edit this page by hand: it is written by `tools/gen_support_page.py`, and a test fails when the
two disagree. Change {{py:mod}}`claveles.support` and re-run that script instead.
```

For what each operation *does*, see {{ref}}`Gates <sec:gates>`, which gives every one of them with its
definition and its Heisenberg action.

(sec:support-operations)=

## Operations

A **representation** is how a program is expressed: {{py:mod}}`~claveles.circuit` is what the simulator
runs, while {{py:mod}}`~claveles.graph` and {{py:mod}}`~claveles.machinery` are what the machine runs, so an
operation is executable on the machine exactly when it has both of those. A **state representation**
is how the simulator holds the state the operation acts on.

The last column says why an operation is missing from a column; it is empty for the operations that
run everywhere.

| Operation | {op_header} | Why restricted |
| --- | {op_rule} | --- |
{chr(10).join(_operation_rows())}

(sec:support-states)=

## States

| State | Held as | Simulator | Machine | Note |
| --- | --- | --- | --- | --- |
{chr(10).join(_state_rows())}

A mode left with no state placed on it is supplied by the machine as its own squeezed state, and is
refused by the simulator, which has no squeezing level to invent.

A circuit may mix the two representations across its modes. Which one a run ends in is settled by
{{py:class}}`~claveles.client.SimulatorClient`'s
{{py:attr}}`~claveles.client.simulator_client.SimulatorClient.state_representation`: `auto` (the default)
runs a circuit in whichever representation it needs, converting into the Fock basis before the first
operation where the circuit requires it anywhere; `bosonic` refuses such a circuit instead of
converting; and `fock` converts every bosonic state up front even where `auto` would have stayed
bosonic. A mode is converted at the cutoff its state was placed with, or at a dimension the simulator
picks where none was chosen, and the conversion truncates the state's infinite Fock tail, so the
weight lost that way is left in the state rather than normalized away.

{fock_limit}

(sec:support-backends)=

## Backends

| State representation | {backend_header} |
| --- | {backend_rule} |
{chr(10).join(_backend_rows())}

`single-threaded` marks a combination that runs, but not as asked: the auto-selecting CPU backend
runs a Fock circuit on one thread, because ITensor is not safe to call concurrently. Asking for a
multi-threaded CPU backend by name is refused instead, so that no one is left expecting parallelism
that cannot be delivered.

The row that applies is the representation a run *ends* in, not the one it starts in: a backend is
chosen once, so a circuit that starts bosonic and reaches the Fock basis partway through is bound
by the `fock` row for the whole of it, and cannot use a GPU backend.

Every backend in this table is reachable through {{py:class}}`~claveles.client.SimulatorClient`, whose
{{py:attr}}`~claveles.client.simulator_client.SimulatorClient.backend` option also takes `cpu` as an alias
for `multi_cpu`. The peak- and shot-level CPU backends are distinct from `multi_cpu`, which chooses
between them.

(sec:support-features)=

## Options

An option that is not an operation in its own right can be restricted in the same way. These are the
ones that are, with the representations they may be used in:

| Option | Representations |
| --- | --- |
{feature_rows}

{{py:attr}}`~claveles.circuit.Operation.post_select` is the parameter that fixes a measurement's outcome
instead of sampling one: a quadrature value on
{{py:class}}`std.MeasureHomodyne <claveles.circuit.ops.std.MeasureHomodyne>`, a photon number on
{{py:class}}`std.MeasureFock <claveles.circuit.ops.std.MeasureFock>`. A circuit that asks for it cannot be
converted to a graph, which has no field to carry it.
"""


def main(argv: list[str] | None = None) -> int:
    """Write or check the support page.

    Args:
        argv: Command line arguments, defaulting to ``sys.argv``.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report whether the page is up to date, without writing")
    args = parser.parse_args(argv)

    rendered = render()
    if args.check:
        current = PAGE.read_text(encoding="utf-8") if PAGE.exists() else ""
        if current == rendered:
            print(f"up to date  {PAGE}")
            return 0
        print(f"OUT OF DATE {PAGE}; re-run tools/gen_support_page.py")
        return 1
    PAGE.write_text(rendered, encoding="utf-8")
    print(f"wrote       {PAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
