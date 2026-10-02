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

"""Decode a circuit ``QuantumProgramResult`` proto, independent of how it was produced.

Both the local Claveles simulator binding and the Cloud Simulator Server return the exact same
``QuantumProgramResult`` proto bytes for a circuit run, so the decode logic lives here rather than
in :mod:`claveles.client._local_simulator_claveles`, which requires the ``claveles_simulator`` wheel at import
time and would make it unusable from a remote client.
"""

from __future__ import annotations

from claveles.circuit.result import CircuitResult, CircuitShotMeasuredValue
from claveles.circuit.state import QuantumState, construct_quantum_state_from_proto
from claveles.client.simulator_client import ShotResult, StateSavePolicy
from claveles.pb.claveles_cloud.program.v1.quantum_program_pb2 import QuantumProgramResult


def _states_by_shot(saved_protos: list, state_save_policy: StateSavePolicy) -> dict[int, QuantumState]:
    """Return the saved state of each shot that has one, keyed by shot index."""
    if state_save_policy == "none":
        return {}
    kept = saved_protos if state_save_policy == "all" else saved_protos[:1]
    return {shot: construct_quantum_state_from_proto(proto) for shot, proto in enumerate(kept)}


def decode_quantum_program_result(
    pb_result_bytes: bytes, state_save_policy: StateSavePolicy, n_shots: int
) -> tuple[CircuitResult, list[ShotResult]]:
    """Parse a serialized ``QuantumProgramResult`` into SDK-level circuit result objects.

    Args:
        pb_result_bytes: Serialized ``QuantumProgramResult`` proto bytes, as returned by
            ``simulate_circuit`` locally or decoded from a Cloud Simulator Server
            ``ResultEnvelope.payload``.
        state_save_policy: One of ``all`` / ``first_only`` / ``none``. Used to select which saved
            states to keep (the proto itself does not record which policy produced it).
        n_shots: Number of shots that were requested; used to build one :class:`ShotResult` per shot.

    Returns:
        A ``(circuit_result, shot_results)`` tuple.
    """
    pb_result = QuantumProgramResult()
    pb_result.ParseFromString(pb_result_bytes)

    # A run is exclusively bosonic or Fock, so only one of these proto fields is populated. Pure and
    # mixed Fock states share circuit_fock_quantum_state, so shot order is preserved either way.
    saved_protos: list = list(pb_result.circuit_state) + list(pb_result.circuit_fock_quantum_state)
    states = _states_by_shot(saved_protos, state_save_policy)
    truncation_errors = list(pb_result.circuit_truncation_error)
    circuit_result = CircuitResult.construct_from_proto(proto_result=pb_result.circuit_result)

    measured_values = list(circuit_result)
    shot_results = [
        ShotResult(
            measured_value=measured_values[shot] if shot < len(measured_values) else CircuitShotMeasuredValue(),
            state=states.get(shot),
            truncation_error=truncation_errors[shot] if shot < len(truncation_errors) else None,
        )
        for shot in range(n_shots)
    ]
    return circuit_result, shot_results
