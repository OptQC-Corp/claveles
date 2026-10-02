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

"""Tests for the saved-state size limit of the local Fock simulator.

Two layers are covered. The first stubs out the native ``simulate_circuit`` binding to assert that
``max_saved_state_elements`` is threaded from :class:`~claveles.client.SimulatorClient` all the way down
to the binding call, which is pure SDK-side wiring. The second runs the compiled simulator and
checks that the limit actually takes effect, since a value that arrives but is misinterpreted would
pass the first layer.

They require the ``claveles_simulator`` package to be importable (the local simulator module imports it
at module load time).
"""

from __future__ import annotations

from typing import Any

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic
from claveles.circuit.state import FockState
from claveles.client import SimulatorClient
from claveles.client.simulator_client import DEFAULT_MAX_SAVED_STATE_ELEMENTS, SimulatorClientResult
from claveles.execute import execute
from claveles.pb.claveles_cloud.program.v1.quantum_program_pb2 import QuantumProgramResult


def _fock_circuit(n_modes: int, cutoff: int) -> CircuitRepr:
    """Build a Fock circuit of @p n_modes modes, each with cutoff @p cutoff.

    Args:
        n_modes: Number of modes.
        cutoff: Fock space dimension of every mode.

    Returns:
        CircuitRepr: The circuit, with a state and one gate per mode.
    """
    circuit = CircuitRepr("fock_saved_state_limit")
    for mode in range(n_modes):
        circuit.Q(mode) | FockState.fock(n=0, cutoff_dim=cutoff)
    for mode in range(n_modes):
        circuit.Q(mode) | intrinsic.PhaseRotation(0.0)
    return circuit


# Wiring: does the value reach the binding?


def _captured_limit(monkeypatch: pytest.MonkeyPatch, client: SimulatorClient) -> Any:  # noqa: ANN401  # whatever the caller passed
    """Run a circuit with a stubbed binding and return the limit it was called with."""
    pytest.importorskip("claveles_simulator")
    import claveles.client._local_simulator_claveles as local_sim  # noqa: PLC0415, PLC2701

    captured = {}

    def _stub(*_args: object, **kwargs: object) -> bytes:
        captured["limit"] = kwargs.get("max_saved_state_elements")
        return QuantumProgramResult().SerializeToString()

    monkeypatch.setattr(local_sim, "simulate_circuit", _stub)
    execute(_fock_circuit(1, 3), client)
    return captured["limit"]


def test_default_limit_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _captured_limit(monkeypatch, SimulatorClient(n_shots=1)) == DEFAULT_MAX_SAVED_STATE_ELEMENTS


def test_explicit_limit_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _captured_limit(monkeypatch, SimulatorClient(n_shots=1, max_saved_state_elements=1234)) == 1234


def test_none_limit_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    # None must survive as None: it lifts the guard rather than selecting the default.
    assert _captured_limit(monkeypatch, SimulatorClient(n_shots=1, max_saved_state_elements=None)) is None


def test_attribute_is_settable_after_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SimulatorClient(n_shots=1)
    client.max_saved_state_elements = 77
    assert _captured_limit(monkeypatch, client) == 77


# Effect: does the limit actually change what the simulator accepts?


def _saved_element_count(client: SimulatorClient, circuit: CircuitRepr) -> int:
    client_result = execute(circuit, client).client_result
    assert isinstance(client_result, SimulatorClientResult)
    saved = client_result.shot_results[0].state
    assert saved is not None
    state, _modes = saved.states[0]
    assert isinstance(state, FockState)
    return int(state.state_vector().size if state.is_pure else state.density_matrix().size)


def test_limit_below_state_size_is_rejected() -> None:
    pytest.importorskip("claveles_simulator")
    # Two modes of cutoff 4 need 16 elements, so a limit of 8 must not admit them.
    client = SimulatorClient(n_shots=1, state_save_policy="all", max_saved_state_elements=8)
    with pytest.raises(RuntimeError, match="exceeds the 8-element limit"):
        execute(_fock_circuit(2, 4), client)


def test_limit_at_state_size_is_accepted() -> None:
    pytest.importorskip("claveles_simulator")
    client = SimulatorClient(n_shots=1, state_save_policy="all", max_saved_state_elements=16)
    assert _saved_element_count(client, _fock_circuit(2, 4)) == 16


def test_raising_the_limit_admits_a_state_the_default_rejects() -> None:
    pytest.importorskip("claveles_simulator")
    # Eleven modes of cutoff 4 need 4^11 elements, four times the default bound.
    circuit = _fock_circuit(11, 4)
    n_elements = 4**11

    default_client = SimulatorClient(n_shots=1, state_save_policy="all")
    with pytest.raises(RuntimeError, match=r"exceeds the \d+-element limit"):
        execute(circuit, default_client)

    raised_client = SimulatorClient(n_shots=1, state_save_policy="all", max_saved_state_elements=n_elements)
    assert _saved_element_count(raised_client, circuit) == n_elements


def test_none_lifts_the_limit() -> None:
    pytest.importorskip("claveles_simulator")
    circuit = _fock_circuit(11, 4)
    client = SimulatorClient(n_shots=1, state_save_policy="all", max_saved_state_elements=None)
    assert _saved_element_count(client, circuit) == 4**11


def test_limit_is_not_consulted_when_no_state_is_saved() -> None:
    pytest.importorskip("claveles_simulator")
    # The guard bounds what gets expanded into a dense array, so a policy that saves nothing has
    # nothing to reject even at a limit far below the state's size.
    client = SimulatorClient(n_shots=1, state_save_policy="none", max_saved_state_elements=1)
    client_result = execute(_fock_circuit(3, 4), client).client_result
    assert isinstance(client_result, SimulatorClientResult)
    assert [shot.state for shot in client_result.shot_results] == [None]
