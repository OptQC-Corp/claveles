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

"""Reference circuits whose simulation results are pinned as golden values.

Each case fixes a representation, a way to build it, and the full set of simulator settings it is
run with. Nothing here depends on the golden files themselves, so the same definitions drive both
the capture script and the comparison test: a case can never be captured under settings that differ
from the ones it is later checked against.

The cases are chosen to cover the paths whose numerics would otherwise go unwatched: both state
representations, pure and mixed Fock states, the two ways a Fock state becomes mixed (a loss channel
and a partial trace), feedforward, post-selected measurement in both representations, MPS truncation,
and the graph and machinery routes.

``backend`` is pinned to ``single_cpu`` for every case. The multi-threaded backend was verified to
give bit-identical results, so this is only to keep the recorded settings unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import TYPE_CHECKING, Any, Literal

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic, std
from claveles.circuit.state import BosonicState, FockState, QuantumState
from claveles.feedforward import feedforward
from claveles.graph.convert import GreedyConverter, GreedyConvertSettings
from claveles.machinery import MachineryRepr

if TYPE_CHECKING:
    from collections.abc import Callable

    from claveles.graph import GraphRepr

SEED = 12345
"""Random seed every case is captured with.

The simulator derives each shot's stream from the seed and the shot index, so a case that gains or
loses an operation does not shift the streams of the other shots.
"""

N_SHOTS = 8
"""Shots per case.

Small on purpose: these cases pin exact numbers rather than estimate distributions, so extra shots
only grow the golden files.
"""

Route = Literal["circuit", "graph", "machinery"]
"""Which representation is submitted to the client."""


@dataclass(frozen=True)
class Case:
    """One reference circuit together with the settings it is simulated under."""

    name: str
    """Identifier, also the stem of the golden file names."""

    route: Route
    """Representation submitted to the client."""

    build: Callable[[], Any]
    """Build the representation. Called fresh each time, because an operation instance cannot be
    applied twice and a circuit therefore cannot be reused."""

    client_kwargs: dict[str, Any] = field(default_factory=dict)
    """Simulator settings beyond the shared defaults, recorded alongside the values."""

    sampling: bool = True
    """Whether the measured values come from sampling rather than post-selection.

    Recorded in the golden metadata as a reading aid: if a sampled case stops matching after a change
    that alters how many random draws a shot consumes, the mismatch may be that change showing
    through rather than a defect. A post-selected case has no such excuse.
    """

    note: str = ""
    """What this case is here to watch."""


def _bosonic_single_mode() -> CircuitRepr:
    circuit = CircuitRepr("golden_bosonic_single_mode")
    circuit.Q(0) | BosonicState.squeezed(r=0.5, phi=0.0)
    circuit.Q(0) | std.MeasureHomodyne(0.0)
    return circuit


@feedforward
def _double(x: float) -> float:
    return 2.0 * x


def _bosonic_multi_mode_ff() -> CircuitRepr:
    circuit = CircuitRepr("golden_bosonic_multi_mode_ff")
    for mode in (0, 1):
        circuit.Q(mode) | BosonicState.squeezed(r=0.5, phi=0.0)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    circuit.Q(0, 1) | std.ControlledZ(g=0.5)
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)
    circuit.Q(1) | std.Displacement(_double(measured))
    return circuit


def _fock_pure_gkp_kerr() -> CircuitRepr:
    # The non-Gaussian state stays on mode 0 and only mode 1 is measured, so the saved state is the
    # interesting one. Measuring mode 0 would leave the vacuum that is reassigned afterwards.
    circuit = CircuitRepr("golden_fock_pure_gkp_kerr")
    circuit.Q(0) | FockState.gkp(cutoff_dim=20)
    circuit.Q(1) | FockState.vacuum(cutoff_dim=20)
    circuit.Q(0) | std.Kerr(kappa=0.1)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    circuit.Q(1) | std.MeasureHomodyne(0.0)
    return circuit


def _fock_mixed_loss() -> CircuitRepr:
    circuit = CircuitRepr("golden_fock_mixed_loss")
    circuit.Q(0) | FockState.fock(n=2, cutoff_dim=8)
    circuit.Q(0) | std.LossChannel(transmittance=0.8)
    return circuit


def _fock_mixed_partial_trace() -> CircuitRepr:
    circuit = CircuitRepr("golden_fock_mixed_partial_trace")
    for mode in (0, 1):
        circuit.Q(mode) | FockState.fock(n=1, cutoff_dim=6)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    circuit.Q(1) | std.PartialTrace()
    circuit.Q(0) | intrinsic.PhaseRotation(0.3)
    return circuit


def _fock_truncated() -> CircuitRepr:
    circuit = CircuitRepr("golden_fock_truncated")
    circuit.Q(0) | FockState.gkp(cutoff_dim=12)
    circuit.Q(1) | FockState.fock(n=1, cutoff_dim=12)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    circuit.Q(0, 1) | std.ControlledZ(g=0.4)
    return circuit


def _post_select_homodyne() -> CircuitRepr:
    circuit = CircuitRepr("golden_post_select_homodyne")
    for mode in (0, 1):
        circuit.Q(mode) | BosonicState.squeezed(r=0.5, phi=0.0)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    circuit.Q(0) | std.MeasureHomodyne(0.0, post_select=0.3)
    return circuit


def _post_select_fock() -> CircuitRepr:
    # Two single photons on a balanced beam splitter interfere, so only the two-photon outcomes carry
    # amplitude. Post-selecting one photon has exactly zero probability and is rejected.
    circuit = CircuitRepr("golden_post_select_fock")
    for mode in (0, 1):
        circuit.Q(mode) | FockState.fock(n=1, cutoff_dim=6)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    circuit.Q(0) | std.MeasureFock(post_select=2)
    return circuit


def _bosonic_epr() -> CircuitRepr:
    # One state covering both modes, which is what a correlation between them needs: placing two
    # marginals instead would lose it. Measured on both modes so that the correlation shows up in the
    # values as well as in the saved state.
    circuit = CircuitRepr("golden_bosonic_epr")
    circuit.Q(0, 1) | QuantumState.epr(r=0.8)
    circuit.Q(0) | std.MeasureHomodyne(0.0)
    return circuit


def _fock_joint_pure() -> CircuitRepr:
    # (|0,1> + |1,0>)/sqrt(2) placed as one state: the photon numbers are anti-correlated, which two
    # marginals could not express. A gate afterwards checks the prepared chain is a usable MPS.
    import numpy as np  # noqa: PLC0415

    sv = np.zeros((3, 3), dtype=np.complex128)
    sv[0, 1] = sv[1, 0] = np.sqrt(0.5)
    circuit = CircuitRepr("golden_fock_joint_pure")
    circuit.Q(0, 1) | QuantumState([FockState(cutoff_dims=[3, 3], state_vector=sv)], [[0, 1]])
    circuit.Q(0) | std.Kerr(kappa=0.1)
    return circuit


def _fock_joint_mixed() -> CircuitRepr:
    # An even mixture of |0,1> and |1,0>: mixed and correlated, so it goes in as one density matrix and
    # comes back purified on an ancilla.
    import numpy as np  # noqa: PLC0415

    dm = np.zeros((4, 4), dtype=np.complex128)
    dm[1, 1] = dm[2, 2] = 0.5
    circuit = CircuitRepr("golden_fock_joint_mixed")
    circuit.Q(0, 1) | QuantumState([FockState(cutoff_dims=[2, 2], density_matrix=dm)], [[0, 1]])
    circuit.Q(0) | intrinsic.PhaseRotation(0.3)
    return circuit


def _route_circuit() -> CircuitRepr:
    circuit = CircuitRepr("golden_route")
    circuit.Q(0) | std.PhaseRotation(pi / 4)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    circuit.Q(0) | std.MeasureHomodyne(0.0)
    circuit.Q(1) | std.MeasureHomodyne(pi / 2)
    return circuit


def _graph_route() -> GraphRepr:
    return GreedyConverter(GreedyConvertSettings(n_local_macronodes=4)).convert(_route_circuit())


def _machinery_route() -> MachineryRepr:
    return MachineryRepr.from_graph_repr(_graph_route())


CASES: tuple[Case, ...] = (
    Case(
        name="bosonic_single_mode",
        route="circuit",
        build=_bosonic_single_mode,
        client_kwargs={"state_save_policy": "all"},
        note="Homodyne sampling from a squeezed state, and the vacuum reassigned to the measured mode.",
    ),
    Case(
        name="bosonic_multi_mode_ff",
        route="circuit",
        build=_bosonic_multi_mode_ff,
        client_kwargs={"state_save_policy": "all"},
        note="Two-mode entangling gates plus a feedforward from a measurement into a displacement.",
    ),
    Case(
        name="fock_pure_gkp_kerr",
        route="circuit",
        build=_fock_pure_gkp_kerr,
        client_kwargs={"state_save_policy": "all"},
        note="Pure multi-mode Fock state carrying a non-Gaussian gate, kept unmeasured on mode 0.",
    ),
    Case(
        name="fock_mixed_loss",
        route="circuit",
        build=_fock_mixed_loss,
        client_kwargs={"state_save_policy": "all"},
        sampling=False,
        note="A loss channel turns the state mixed, so the result is a density matrix.",
    ),
    Case(
        name="fock_mixed_partial_trace",
        route="circuit",
        build=_fock_mixed_partial_trace,
        client_kwargs={"state_save_policy": "all"},
        sampling=False,
        note="The other route to a mixed state: a partial trace followed by a further gate.",
    ),
    Case(
        name="fock_truncated",
        route="circuit",
        build=_fock_truncated,
        client_kwargs={
            "state_save_policy": "all",
            "max_bond_dim": 4,
            "svd_cutoff": 1e-6,
            "track_truncation_error": True,
        },
        sampling=False,
        note="MPS truncation: a bond dimension low enough to discard weight, with the error tracked.",
    ),
    Case(
        name="post_select_homodyne",
        route="circuit",
        build=_post_select_homodyne,
        client_kwargs={"state_save_policy": "all"},
        sampling=False,
        note="Post-selected homodyne outcome and the projected bosonic state it leaves.",
    ),
    Case(
        name="post_select_fock",
        route="circuit",
        build=_post_select_fock,
        client_kwargs={"state_save_policy": "all"},
        sampling=False,
        note="Post-selected Fock outcome, on the two-photon interference the balanced splitter allows.",
    ),
    Case(
        name="bosonic_epr",
        route="circuit",
        build=_bosonic_epr,
        client_kwargs={"state_save_policy": "all"},
        note="A two-mode state placed as one state, exercising the explicit mode indices on the wire.",
    ),
    Case(
        name="fock_joint_pure",
        route="circuit",
        build=_fock_joint_pure,
        client_kwargs={"state_save_policy": "all"},
        sampling=False,
        note="A pure Fock state spanning two modes, prepared through the joint-state path.",
    ),
    Case(
        name="fock_joint_mixed",
        route="circuit",
        build=_fock_joint_mixed,
        client_kwargs={"state_save_policy": "all"},
        sampling=False,
        note="A mixed Fock state spanning two modes, purified onto an ancilla on the way in.",
    ),
    Case(
        name="graph_route",
        route="graph",
        build=_graph_route,
        note="The same circuit embedded into a graph representation and simulated there.",
    ),
    Case(
        name="machinery_route",
        route="machinery",
        build=_machinery_route,
        note="The same graph lowered to a machinery representation and simulated there.",
    ),
)


def case_by_name(name: str) -> Case:
    """Look up a case by name.

    Args:
        name: Case name.

    Returns:
        Case: The matching case.

    Raises:
        KeyError: If no case has that name.
    """
    for case in CASES:
        if case.name == name:
            return case
    msg = f"Unknown golden case {name!r}. Known cases: {sorted(c.name for c in CASES)}"
    raise KeyError(msg)
