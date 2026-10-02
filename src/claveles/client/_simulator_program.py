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

"""Serialize/decode logic shared by every transport that reaches a Cloud Simulator Server (CSS).

``mode="simulator-server"`` (:mod:`claveles.client._simulator_server`) reaches CSS directly over its own
``/v1/jobs`` REST API; ``mode="remote"`` (:mod:`claveles.client._simulator_remote`) reaches the very same
CSS through the Cloud User API and the ``simulator-bridge`` service instead. Both hand CSS the same
serialized program bytes and get back the same ``ResultEnvelope.payload`` bytes, so the code that turns
an SDK representation into those bytes and back lives here once rather than once per transport.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from claveles.circuit.program import CircuitRepr
from claveles.client._circuit_result_decode import decode_quantum_program_result
from claveles.client._state_representation import (
    reject_unusable_initial_states,
    report_unchosen_conversion_cutoffs,
    resolve_for_circuit,
)
from claveles.client.abstract import representation_kind
from claveles.graph.result import GraphResult
from claveles.machinery.result import MachineryResult
from claveles.pb.claveles_cloud.program.v1.graph_pb2 import GraphResult as PbGraphResult
from claveles.pb.claveles_cloud.program.v1.machinery_pb2 import MachineryResult as PbMachineryResult

if TYPE_CHECKING:
    from claveles.client.abstract import ResultType
    from claveles.client.simulator_client import ShotResult
    from claveles.graph.program import GraphRepr
    from claveles.machinery.program import MachineryRepr
    from claveles.support import StateRepresentationOption

POLLING_TIMEOUT_MARGIN_MSEC = 600_000
"""Extra margin (ms) added on top of timeout_msec to derive the client-side polling deadline."""

DEFAULT_SERVER_TIMEOUT_MSEC = 3_600_000
"""CSS's own *documented* default execution timeout (api-design.md), used to size the client-side
deadline when the caller leaves timeout_msec unset.

Known limitation: CSS has no way to report its actually-configured default (settable per-instance
via ``CLAVELES_SERVER_DEFAULT_TIMEOUT_MSEC``) back to a caller, so this constant can drift from
reality. An operator raising that env var past this value would make the client raise
``TimeoutError`` before a job that CSS itself would still let finish. Fixing this for real needs a
wire-protocol change (CSS exposing its configured default); not done here. Read by both transports,
since both wait on the same CSS-side default."""

TERMINAL_STATUSES = frozenset({"SUCCESS", "FAILURE", "TIMEOUT"})
"""Statuses a job does not leave. Shared because both transports poll the same underlying CSS job,
even though the Cloud User API additionally reports a non-terminal ``SUBMITTED`` phase (before the
bridge has picked the job up) that CSS itself never reports."""


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse a ``JobTimeline``-shaped timestamp, which may carry a trailing ``Z`` UTC designator.

    Args:
        value: An ISO-8601 timestamp string, or ``None``.

    Returns:
        The parsed timestamp, or ``None`` if ``value`` was ``None``.
    """
    if value is None:
        return None
    # `datetime.fromisoformat` only accepts a bare "Z" from Python 3.11 onward, but this package
    # supports 3.10, so the "+00:00" replacement below is load-bearing, not redundant.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162


def serialize_program(
    representation: CircuitRepr | GraphRepr | MachineryRepr,
    *,
    decompose_std: bool,
    state_representation: StateRepresentationOption,
) -> tuple[str, bytes]:
    """Serialize a representation for submission to a Cloud Simulator Server.

    Args:
        representation: Representation to submit.
        decompose_std: Whether to lower std operations to their intrinsic decomposition before
            serialization. Only meaningful for circuits.
        state_representation: Which representation the caller asked the circuit to be run in. Only
            meaningful for circuits, and settled here rather than on the server so that a circuit
            refused for asking the impossible is refused before it is uploaded.

    Returns:
        A ``(program_type, program_bytes)`` tuple, where ``program_bytes`` is the raw serialized
        proto. Callers that need it base64-encoded for a JSON envelope (``mode="simulator-server"``)
        encode it themselves; callers that upload it as a raw S3 object (``mode="remote"``) do not.

    Raises:
        TypeError: If ``representation`` is not a ``CircuitRepr``/``GraphRepr``/``MachineryRepr``.
    """  # noqa: DOC502  # TypeError comes from `representation_kind`, which is what refuses the value
    # CSS names a program type with the upper-cased representation name, so the name the SDK already
    # has for each representation is the wire tag, rather than a second three-way branch beside it.
    program_type = representation_kind(representation).upper()

    if isinstance(representation, CircuitRepr):
        reject_unusable_initial_states(representation)
        pb_circuit = representation.proto(decompose_std=decompose_std)
        if resolve_for_circuit(representation, state_representation) == "fock":
            report_unchosen_conversion_cutoffs(pb_circuit)
        return program_type, pb_circuit.SerializeToString()

    # Graph and machinery serialize through the `proto()` every representation has (`ProgramRepr`),
    # and take none of the circuit-only options above.
    return program_type, representation.proto().SerializeToString()


def decode_result_payload(
    program_type: str, payload: bytes, n_shots: int
) -> tuple[ResultType, list[ShotResult] | None]:
    """Decode a ``ResultEnvelope.payload`` into the representation-appropriate result type.

    Only ``CIRCUIT`` results carry shot results (api-design.md S-6: the binding's
    ``simulate_circuit``/``simulate_graph``/``simulate_machinery`` return value asymmetry --
    ``QuantumProgramResult`` vs. the result proto directly).

    Returns:
        A ``(result, shot_results)`` tuple. ``result`` is the one result the program's representation
        produces, rather than three optional ones of which the caller has to find the populated one;
        ``shot_results`` is ``None`` for graph/machinery, which is how ``mode="local"`` reports the same
        absence.

    Raises:
        ValueError: If ``program_type`` is not one of ``CIRCUIT``/``GRAPH``/``MACHINERY``.
    """
    if program_type == "CIRCUIT":
        # The policy is not a client option here: CSS fixes it, whether reached directly or through
        # the Cloud User API, and the constant is what `SimulatorClient.state_save_policy` reports
        # for both such modes, so the two cannot drift apart.
        from claveles.client.simulator_client import (  # noqa: PLC0415  # avoids a cycle
            SIMULATOR_SERVER_STATE_SAVE_POLICY,
        )

        circuit_result, shot_results = decode_quantum_program_result(
            payload, SIMULATOR_SERVER_STATE_SAVE_POLICY, n_shots
        )
        return circuit_result, shot_results
    if program_type == "GRAPH":
        pb_graph_result = PbGraphResult()
        pb_graph_result.ParseFromString(payload)
        return GraphResult.construct_from_proto(proto_result=pb_graph_result), None
    if program_type == "MACHINERY":
        pb_machinery_result = PbMachineryResult()
        pb_machinery_result.ParseFromString(payload)
        return MachineryResult.construct_from_proto(proto_result=pb_machinery_result), None
    msg = f"Unsupported program_type: {program_type!r}"
    raise ValueError(msg)
