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

"""Running a golden case and reading or writing its recorded result.

A record is stored as two files per case: a ``.json`` holding the settings the case was captured
under together with the structural facts about each saved state, and a ``.npz`` holding the numbers.
The split keeps the settings reviewable in a diff while the arrays stay in exact binary form.

Exactness matters here. Everything crosses the protobuf boundary as 32-bit floats -- measured values,
Gaussian means and covariances, Fock amplitudes and density matrix entries alike -- so a captured
value is already rounded to float32 before it reaches this module. Storing it as float64 therefore
round-trips it without loss, and a re-run of an unchanged implementation reproduces it bit for bit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from claveles.circuit.state import BasicState, BosonicState, FockState
from claveles.client import SimulatorClient

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from golden.cases import Case

DATA_DIR = Path(__file__).parent / "data"
"""Directory holding the ``.json`` / ``.npz`` pair of each case."""

MEASURED_COLUMNS: dict[str, tuple[str, ...]] = {
    # The circuit route reads out a mode; the other two read out a macronode, which is what their
    # `index` column holds.
    "circuit": ("shot", "mode", "value"),
    "graph": ("shot", "index", "h", "w", "m_b", "m_d"),
    "machinery": ("shot", "index", "m_a", "m_b", "m_c", "m_d"),
}
"""Column layout of the measured-value table, per representation.

Each representation reads out a different tuple, so the table is stored as a plain 2-D array plus the
names of its columns rather than as one struct per representation. The leading ``shot`` column makes
the table self-describing even when shots contribute different numbers of rows.
"""


@dataclass(frozen=True)
class StateRecord:
    """One saved state, split into structural facts and numeric arrays."""

    kind: str
    """``bosonic``, ``fock``, or ``zero_mode`` for a state whose every mode was measured away."""

    meta: dict[str, Any]
    """Structure of the state: mode indices, and peak count or cutoff dimensions and purity."""

    arrays: dict[str, NDArray[Any]]
    """The numbers. ``coeffs`` / ``means`` / ``covs`` for bosonic, ``state_vector`` or
    ``density_matrix`` for Fock."""


@dataclass(frozen=True)
class GoldenRecord:
    """Everything pinned for one case."""

    case: str
    route: str
    conditions: dict[str, Any]
    """Every simulator setting the case was run with, so it can be captured again."""

    measured: NDArray[np.float64]
    """Measured values, laid out per :data:`MEASURED_COLUMNS`."""

    states: list[StateRecord] = field(default_factory=list)
    truncation_errors: list[float] = field(default_factory=list)
    sampling: bool = True
    note: str = ""


def _conditions(case: Case) -> dict[str, Any]:
    """Collect the full set of settings a case runs under, defaults included."""
    from golden.cases import N_SHOTS, SEED  # noqa: PLC0415  # avoids a circular import at module load

    conditions = {
        "n_shots": N_SHOTS,
        "random_seed": SEED,
        "backend": "single_cpu",
        "state_save_policy": "none",
        "resource_squeezing_level": 10.0,
        "max_bond_dim": None,
        "svd_cutoff": 0.0,
        "track_truncation_error": False,
        "decompose_std": False,
    }
    conditions.update(case.client_kwargs)
    return conditions


def _state_record(state: BasicState, modes: list[int]) -> StateRecord:
    """Normalize a saved state into a record.

    Args:
        state: One component of a state returned by the simulator.
        modes: The modes that component occupies.

    Returns:
        StateRecord: The normalized state.

    Raises:
        TypeError: If the state is of an unexpected type.
    """
    if isinstance(state, BosonicState):
        return StateRecord(
            kind="bosonic",
            meta={"modes": modes, "n_modes": state.n_modes, "n_peaks": state.n_peaks},
            arrays={
                "coeffs": np.asarray(state.coeffs, dtype=np.complex128),
                "means": np.stack([g.mean for g in state.gaussian_states]).astype(np.complex128),
                "covs": np.stack([g.cov for g in state.gaussian_states]).astype(np.float64),
            },
        )
    if isinstance(state, FockState):
        arrays = (
            {"state_vector": np.asarray(state.state_vector(), dtype=np.complex128)}
            if state.is_pure
            else {"density_matrix": np.asarray(state.density_matrix(), dtype=np.complex128)}
        )
        return StateRecord(
            kind="fock",
            meta={
                "modes": modes,
                "n_modes": state.n_modes,
                "cutoff_dims": state.cutoff_dims,
                "is_pure": state.is_pure,
            },
            arrays=arrays,
        )
    msg = f"Unexpected saved-state type: {type(state).__name__}"
    raise TypeError(msg)


def _measured_table(route: str, client_result) -> NDArray[np.float64]:  # noqa: ANN001  # one of three result types
    """Flatten the measured values of every shot into the table described by :data:`MEASURED_COLUMNS`."""
    rows: list[list[float]] = []
    if route == "circuit":
        for shot_index, shot in enumerate(client_result.circuit_result):
            rows.extend([float(shot_index), float(mv.mode), float(mv.value)] for mv in shot)
    elif route == "graph":
        for shot_index, shot in enumerate(client_result.graph_result):
            rows.extend(
                [float(shot_index), float(mv.index), float(mv.h), float(mv.w), float(mv.m_b), float(mv.m_d)]
                for mv in shot
            )
    else:
        for shot_index, shot in enumerate(client_result.machinery_result):
            rows.extend(
                [float(shot_index), float(mv.index), float(mv.m_a), float(mv.m_b), float(mv.m_c), float(mv.m_d)]
                for mv in shot
            )
    return np.array(rows, dtype=np.float64).reshape(len(rows), len(MEASURED_COLUMNS[route]))


def run_case(case: Case) -> GoldenRecord:
    """Simulate a case and normalize the result into a record.

    Args:
        case: The case to run.

    Returns:
        GoldenRecord: The result, ready to be saved or compared.
    """
    conditions = _conditions(case)
    # The record keeps every setting so the run can be reproduced from it, but the client is given only
    # the ones the route reads: `resource_squeezing_level` belongs to the graph and machinery routes, and
    # handing it to a circuit run reports an IgnoredOptionWarning, which is exactly right for a caller
    # and only noise for a harness that spells its defaults out on purpose.
    client_kwargs = {name: value for name, value in conditions.items() if name != "resource_squeezing_level"}
    if case.route != "circuit":
        client_kwargs["resource_squeezing_level"] = conditions["resource_squeezing_level"]
    client = SimulatorClient(**client_kwargs)
    # Submitted through the client rather than through execute(), which only adds wall-clock timing
    # around the same call and widens the result type.
    client_result = client.run(case.build())

    # Only the circuit route pairs states with shots; graph and machinery report measured values only.
    shot_results = client_result.shot_results if case.route == "circuit" else []

    # One component per saved shot, so the container's single entry is what gets recorded. A shot that
    # measured every mode saves a container with no components at all, which is recorded as such so it
    # stays distinguishable from no state having been saved. Shots the save policy skipped contribute
    # nothing, which is how the record stays the same length as the set of saved states.
    states: list[StateRecord] = []
    for shot in shot_results:
        if shot.state is None:
            continue
        if not shot.state.states:
            states.append(StateRecord(kind="zero_mode", meta={"n_modes": 0, "modes": []}, arrays={}))
            continue
        states.extend(starmap(_state_record, shot.state.states))
    return GoldenRecord(
        case=case.name,
        route=case.route,
        conditions=conditions,
        measured=_measured_table(case.route, client_result),
        states=states,
        truncation_errors=[float(shot.truncation_error) for shot in shot_results if shot.truncation_error is not None],
        sampling=case.sampling,
        note=case.note,
    )


def _array_key(state_index: int, name: str) -> str:
    return f"state.{state_index}.{name}"


def save(record: GoldenRecord, directory: Path) -> None:
    """Write a record as a ``.json`` / ``.npz`` pair.

    Args:
        record: The record to write.
        directory: Directory to write into. Created if missing.
    """
    directory.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, NDArray[Any]] = {"measured": record.measured}
    for state_index, state in enumerate(record.states):
        for name, array in state.arrays.items():
            arrays[_array_key(state_index, name)] = array
    # The keys are array names built here, never numpy's own `allow_pickle` keyword, which is what
    # a checker has to assume a **kwargs unpack could supply.
    np.savez_compressed(directory / f"{record.case}.npz", **arrays)  # ty: ignore[invalid-argument-type]

    manifest = {
        "case": record.case,
        "route": record.route,
        "note": record.note,
        "sampling": record.sampling,
        "conditions": record.conditions,
        "measured_columns": list(MEASURED_COLUMNS[record.route]),
        "measured_rows": int(record.measured.shape[0]),
        "truncation_errors": record.truncation_errors,
        "states": [
            {"kind": state.kind, "meta": state.meta, "arrays": sorted(state.arrays)} for state in record.states
        ],
    }
    path = directory / f"{record.case}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load(name: str, directory: Path) -> GoldenRecord:
    """Read back a record written by :func:`save`.

    Args:
        name: Case name.
        directory: Directory holding the golden files.

    Returns:
        GoldenRecord: The stored record.
    """
    manifest = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    with np.load(directory / f"{name}.npz") as loaded:
        measured = loaded["measured"]
        states = [
            StateRecord(
                kind=entry["kind"],
                meta=entry["meta"],
                arrays={key: loaded[_array_key(state_index, key)] for key in entry["arrays"]},
            )
            for state_index, entry in enumerate(manifest["states"])
        ]

    return GoldenRecord(
        case=manifest["case"],
        route=manifest["route"],
        conditions=manifest["conditions"],
        measured=measured,
        states=states,
        truncation_errors=[float(err) for err in manifest["truncation_errors"]],
        sampling=manifest["sampling"],
        note=manifest["note"],
    )
