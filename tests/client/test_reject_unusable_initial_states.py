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

"""Tests that both simulator clients refuse modes the simulator cannot start from.

Two cases are refused. A mode nothing placed a state on is left for the backend to supply, and the
circuit simulator has no default to supply. A mode carrying a ``HardwareConstrainedSqueezedState``
names a state whose squeezing level the hardware fixes, which the simulator likewise does not have.
Both are reported up front with the mode numbers rather than surfacing as a degenerate state further
down.

The check lives in ``claveles.client._state_representation`` so that the local simulator and the Cloud
Simulator Server cannot disagree about it, and both paths are exercised here for that reason. The
local ones require the ``claveles_simulator`` package to be importable (the local simulator module imports
it at module load time); the server ones do not, since serialization happens before any upload.
"""

from __future__ import annotations

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic
from claveles.circuit.state import BosonicState, HardwareConstrainedSqueezedState
from claveles.client._simulator_program import serialize_program as _serialize_program  # noqa: PLC2701
from claveles.pb.claveles_cloud.program.v1.quantum_program_pb2 import QuantumProgramResult


def _open_mode_circuit() -> CircuitRepr:
    """A circuit whose mode 1 is acted on without a state being placed on it."""
    circuit = CircuitRepr("partly_placed")
    circuit.Q(0) | BosonicState.vacuum()
    circuit.Q(2) | BosonicState.vacuum()
    circuit.Q(1) | intrinsic.PhaseRotation(0.0)
    return circuit


def _hardware_state_circuit() -> CircuitRepr:
    """A circuit whose mode 1 carries the state only the machine can supply."""
    circuit = CircuitRepr("hardware_state")
    circuit.Q(0) | BosonicState.vacuum()
    circuit.Q(1) | HardwareConstrainedSqueezedState()
    return circuit


def _closed_circuit() -> CircuitRepr:
    """A circuit with a state on every mode."""
    circuit = CircuitRepr("closed")
    circuit.Q(0) | BosonicState.vacuum()
    circuit.Q(1) | BosonicState.vacuum()
    return circuit


def test_local_run_circuit_rejects_an_open_mode() -> None:
    pytest.importorskip("claveles_simulator")
    import claveles.client._local_simulator_claveles as local_sim  # noqa: PLC0415, PLC2701

    circuit = _open_mode_circuit()

    assert circuit.open_modes == [1]
    assert circuit.input_arity == 1

    with pytest.raises(ValueError, match=r"Mode\(s\) \[1\] have no state placed"):
        local_sim.local_run_circuit(
            backend="single_cpu",
            n_shots=1,
            state_save_policy="all",
            circuit=circuit,
        )


def test_local_run_circuit_rejects_a_hardware_constrained_state() -> None:
    pytest.importorskip("claveles_simulator")
    import claveles.client._local_simulator_claveles as local_sim  # noqa: PLC0415, PLC2701

    with pytest.raises(ValueError, match=r"Mode\(s\) \[1\] carry a HardwareConstrainedSqueezedState"):
        local_sim.local_run_circuit(
            backend="single_cpu",
            n_shots=1,
            state_save_policy="all",
            circuit=_hardware_state_circuit(),
        )


def test_local_run_circuit_accepts_a_closed_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("claveles_simulator")
    import claveles.client._local_simulator_claveles as local_sim  # noqa: PLC0415, PLC2701

    # A circuit with a state on every mode must pass the check; stub the native binding so the test
    # does not depend on the compiled simulator.
    monkeypatch.setattr(local_sim, "simulate_circuit", lambda *_, **__: QuantumProgramResult().SerializeToString())

    circuit = _closed_circuit()

    assert circuit.open_modes == []

    result = local_sim.local_run_circuit(
        backend="single_cpu",
        n_shots=1,
        state_save_policy="all",
        circuit=circuit,
    )
    # The stub returns an empty result, so the shot carries no state; what matters is that the call
    # was not refused, and that the requested shot is reported.
    assert len(result.shot_results) == 1


def test_server_path_rejects_an_open_mode() -> None:
    with pytest.raises(ValueError, match=r"Mode\(s\) \[1\] have no state placed"):
        _serialize_program(_open_mode_circuit(), decompose_std=False, state_representation="auto")


def test_server_path_rejects_a_hardware_constrained_state() -> None:
    with pytest.raises(ValueError, match=r"Mode\(s\) \[1\] carry a HardwareConstrainedSqueezedState"):
        _serialize_program(_hardware_state_circuit(), decompose_std=False, state_representation="auto")


def test_server_path_accepts_a_closed_circuit() -> None:
    program_type, program = _serialize_program(_closed_circuit(), decompose_std=False, state_representation="auto")

    assert program_type == "CIRCUIT"
    assert program


@pytest.mark.parametrize(
    "build_circuit",
    [_open_mode_circuit, _hardware_state_circuit],
    ids=["open_mode", "hardware_constrained_state"],
)
def test_both_paths_refuse_with_the_same_message(build_circuit) -> None:  # noqa: ANN001  # a fixture-free factory
    """The two clients share one check, so neither may word the refusal its own way."""
    pytest.importorskip("claveles_simulator")
    import claveles.client._local_simulator_claveles as local_sim  # noqa: PLC0415, PLC2701

    with pytest.raises(ValueError, match=r"Mode\(s\) \[1\]") as local_error:
        local_sim.local_run_circuit(
            backend="single_cpu",
            n_shots=1,
            state_save_policy="all",
            circuit=build_circuit(),
        )
    with pytest.raises(ValueError, match=r"Mode\(s\) \[1\]") as server_error:
        _serialize_program(build_circuit(), decompose_std=False, state_representation="auto")

    assert str(local_error.value) == str(server_error.value)


@pytest.mark.parametrize("build_circuit", [_open_mode_circuit, _hardware_state_circuit])
def test_the_refusal_names_no_removed_api(build_circuit) -> None:  # noqa: ANN001  # a fixture-free factory
    """The message must not advise APIs the SDK no longer has."""
    with pytest.raises(ValueError, match=r"Mode\(s\) \[1\]") as error:
        _serialize_program(build_circuit(), decompose_std=False, state_representation="auto")

    message = str(error.value)
    assert "set_initial_state" not in message
    assert "FockInitialState" not in message
