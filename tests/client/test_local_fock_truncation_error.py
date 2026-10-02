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

"""Tests for parsing per-shot Fock truncation errors from the local simulator.

These tests stub out the native ``simulate_circuit`` binding and feed a crafted
``QuantumProgramResult`` so the SDK-side parsing can be verified without running
the compiled simulator. The end-to-end behaviour of the truncation itself is
covered in the simulator project's ``tests/circuit`` suite, which runs against the
built wheel. They require the ``claveles_simulator`` package to be importable (the
local simulator module imports it at module load time).
"""

from __future__ import annotations

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.state import FockState
from claveles.pb.claveles_cloud.program.v1.quantum_program_pb2 import QuantumProgramResult


def _crafted_result_bytes(truncation_errors: list[float]) -> bytes:
    """Build a QuantumProgramResult carrying the given per-shot truncation errors.

    Args:
        truncation_errors: One accumulated SVD truncation error per shot.

    Returns:
        bytes: Serialized QuantumProgramResult.
    """
    result = QuantumProgramResult()
    result.circuit_truncation_error.extend(truncation_errors)
    return result.SerializeToString()


def _run(monkeypatch: pytest.MonkeyPatch, truncation_errors: list[float], *, track: bool) -> list[float | None]:
    pytest.importorskip("claveles_simulator")
    import claveles.client._local_simulator_claveles as local_sim  # noqa: PLC0415, PLC2701

    monkeypatch.setattr(local_sim, "simulate_circuit", lambda *_, **__: _crafted_result_bytes(truncation_errors))

    circuit = CircuitRepr("fock_truncation")
    circuit.Q(0)
    circuit.Q(0) | FockState.fock(n=1, cutoff_dim=5)

    result = local_sim.local_run_circuit(
        backend="single_cpu",
        n_shots=len(truncation_errors),
        state_save_policy="none",
        circuit=circuit,
        track_truncation_error=track,
    )
    return [shot.truncation_error for shot in result.shot_results]


def test_local_run_circuit_parses_truncation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    errors = _run(monkeypatch, [0.25, 0.125, 0.0], track=True)
    assert errors == pytest.approx([0.25, 0.125, 0.0])


def test_local_run_circuit_truncation_errors_empty_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # When the proto carries no truncation-error field (tracking disabled), the parsed list is empty.
    errors = _run(monkeypatch, [], track=False)
    assert errors == []
