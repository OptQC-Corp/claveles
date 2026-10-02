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

"""Tests for the per-shot pairing of measured values with the state each shot left behind."""

from __future__ import annotations

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import QuantumState
from claveles.client import SimulatorClient
from claveles.client.simulator_client import SimulatorClientResult, StateSavePolicy
from claveles.execute import execute
from claveles.graph import GraphRepr

_N_SHOTS = 3


def _circuit() -> CircuitRepr:
    """A two-mode circuit that measures one mode and leaves the other carrying a state."""
    circuit = CircuitRepr("shot_results")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.squeezed(r=0.5)
    circuit.Q(0) | std.MeasureHomodyne(0.0)
    return circuit


@pytest.mark.parametrize(
    ("policy", "expected_saved_shots"),
    [("all", [0, 1, 2]), ("first_only", [0]), ("none", [])],
)
def test_which_shots_carry_a_state_follows_the_save_policy(
    policy: StateSavePolicy, expected_saved_shots: list[int]
) -> None:
    pytest.importorskip("claveles_simulator")

    result = SimulatorClient(n_shots=_N_SHOTS, state_save_policy=policy).run(_circuit())

    # Every shot is reported whatever the policy; only whether it carries a state changes.
    assert len(result.shot_results) == _N_SHOTS
    saved = [shot for shot, entry in enumerate(result.shot_results) if entry.state is not None]
    assert saved == expected_saved_shots


def test_a_shot_pairs_its_measured_values_with_its_surviving_modes() -> None:
    pytest.importorskip("claveles_simulator")

    result = SimulatorClient(n_shots=_N_SHOTS, state_save_policy="all").run(_circuit())

    for entry in result.shot_results:
        # Mode 0 was measured and so is gone from the state; mode 1 was not measured and remains.
        assert entry.measured_value.get_value(0) is not None
        assert entry.measured_value.get_value(1) is None
        assert entry.state is not None
        assert entry.state.modes == [1]


def test_the_paired_measured_values_are_the_ones_the_circuit_result_reports() -> None:
    pytest.importorskip("claveles_simulator")

    result = SimulatorClient(n_shots=_N_SHOTS, state_save_policy="all").run(_circuit())

    for shot, entry in enumerate(result.shot_results):
        assert entry.measured_value == result.circuit_result.get_shot_measured_value(shot)


def test_a_shot_that_measured_every_mode_leaves_a_state_with_no_modes() -> None:
    pytest.importorskip("claveles_simulator")

    circuit = CircuitRepr("all_measured")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.MeasureHomodyne(0.0)

    result = SimulatorClient(n_shots=1, state_save_policy="all").run(circuit)

    state = result.shot_results[0].state
    # Distinct from the state not having been saved, which reports None.
    assert state is not None
    assert state.n_modes == 0
    assert state.modes == []


def test_truncation_error_is_none_unless_it_is_tracked() -> None:
    pytest.importorskip("claveles_simulator")

    result = SimulatorClient(n_shots=_N_SHOTS, state_save_policy="none").run(_circuit())

    assert [entry.truncation_error for entry in result.shot_results] == [None] * _N_SHOTS


def test_graph_execution_has_no_shot_results() -> None:
    pytest.importorskip("claveles_simulator")

    graph = GraphRepr(2, 2)
    client_result = execute(graph, SimulatorClient(n_shots=1)).client_result

    assert isinstance(client_result, SimulatorClientResult)
    # Graph and machinery execution report measured values only, so there is nothing to pair.
    with pytest.raises(ValueError, match="only available for circuit execution"):
        _ = client_result.shot_results
