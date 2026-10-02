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

"""Locally run circuit representation using Claveles simulator."""

from __future__ import annotations

try:
    from claveles_simulator._claveles_simulator_impl import (  # type: ignore  # noqa: PGH003, PLC2701  # pybind11 impl module has no public alias
        simulate_circuit,
        simulate_graph,
        simulate_machinery,
    )
except (ImportError, ModuleNotFoundError) as e:
    msg = """Claveles simulator is not installed.
Install it with:

pip install claveles-simulator
"""
    raise ImportError(msg) from e


import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from claveles.circuit.result import CircuitResult, CircuitShotMeasuredValue
from claveles.circuit.state import (
    QuantumState,
    construct_quantum_state_from_proto,
)
from claveles.client._state_representation import (
    reject_unusable_initial_states,
    report_unchosen_conversion_cutoffs,
    resolve_for_circuit,
)
from claveles.client.exceptions import JobFailedError
from claveles.client.simulator_client import DEFAULT_MAX_SAVED_STATE_ELEMENTS, ShotResult, StateSavePolicy
from claveles.graph.result import GraphResult
from claveles.machinery.result import MachineryResult
from claveles.pb.claveles_cloud.program.v1.graph_pb2 import GraphResult as PbGraphResult
from claveles.pb.claveles_cloud.program.v1.machinery_pb2 import MachineryResult as PbMachineryResult
from claveles.pb.claveles_cloud.program.v1.quantum_program_pb2 import QuantumProgramResult
from claveles.support import Backend, StateRepresentationOption, check_backend

if TYPE_CHECKING:
    from collections.abc import Iterator

    from claveles.circuit.program import CircuitRepr
    from claveles.graph.program import GraphRepr
    from claveles.machinery.program import MachineryRepr

logger = logging.getLogger(__name__)


@contextmanager
def _job_failure_reported_uniformly() -> Iterator[None]:
    """Report a failed simulation as :class:`JobFailedError`, whichever mode ran it.

    The binding signals a failed simulation with ``RuntimeError(message, error_id)``, while the Cloud
    Simulator Server path builds the same pair out of the job's error record. Raising one class here
    means a caller writes one ``except`` clause for both, and because that class is itself a
    ``RuntimeError`` carrying the same ``args``, code written against the bare ``RuntimeError`` the
    binding raises keeps working.

    Yields:
        None: The body runs with the binding's failures translated on the way out.

    Raises:
        JobFailedError: If the wrapped call fails.
    """
    try:
        yield
    except JobFailedError:
        raise
    except RuntimeError as e:
        # The pair is read defensively rather than splatted: the binding's message is always first, but
        # nothing guarantees an error id follows it, and a missing one has to stay missing.
        message = str(e.args[0]) if e.args else str(e)
        error_id = e.args[1] if len(e.args) > 1 and isinstance(e.args[1], int) else None
        raise JobFailedError(message, error_id) from e


@dataclass(frozen=True)
class LocalCircuitResult:
    """The result of executing a quantum circuit with the Claveles circuit simulator."""

    execution_time: timedelta
    """The time to execute a quantum circuit."""

    circuit_result: CircuitResult
    """Measurement results after circuit execution."""

    shot_results: list[ShotResult]
    """What each shot produced, in shot order: the values it measured and the state it left."""


def _states_by_shot(saved_protos: list, state_save_policy: StateSavePolicy) -> dict[int, QuantumState]:
    """Return the saved state of each shot that has one, keyed by shot index.

    Args:
        saved_protos: Saved-state messages emitted by the simulator, in shot order.
        state_save_policy: One of ``all`` / ``first_only`` / ``none``.

    Returns:
        dict[int, QuantumState]: The saved state of each shot that was saved.

    Details:
        The simulator emits a message only for the shots the policy asked for, so the mapping onto
        shot indices comes from the policy: ``all`` saves every shot in order, ``first_only`` saves
        shot 0, and ``none`` saves nothing. A state too large to hand back is refused outright rather
        than skipped, so under ``all`` there is no gap to account for.
    """
    if state_save_policy == "none":
        return {}
    kept = saved_protos if state_save_policy == "all" else saved_protos[:1]
    return {shot: construct_quantum_state_from_proto(proto) for shot, proto in enumerate(kept)}


def local_run_circuit(  # noqa: PLR0913  # mirrors the public simulator configuration options
    backend: Backend,
    n_shots: int,
    state_save_policy: StateSavePolicy,
    circuit: CircuitRepr,
    seed: int | None = None,
    *,
    max_bond_dim: int | None = None,
    svd_cutoff: float = 0.0,
    track_truncation_error: bool = False,
    max_saved_state_elements: int | None = DEFAULT_MAX_SAVED_STATE_ELEMENTS,
    decompose_std: bool = False,
    state_representation: StateRepresentationOption = "auto",
) -> LocalCircuitResult:
    reject_unusable_initial_states(circuit)
    # Checked here rather than left to the C++ side: whether a backend can run this circuit is known
    # before anything is serialized, and claveles.support is where the answer is recorded. The
    # representation asked about is the one the run ends in, since a run that converts partway through
    # is bound by the Fock representation's restrictions for the whole of it.
    resolved = resolve_for_circuit(circuit, state_representation)
    check_backend(
        resolved,
        backend,
        [op.name() for op in circuit if not op.is_state_preparation()],
    )

    pb_circuit = circuit.proto(decompose_std=decompose_std)
    if resolved == "fock":
        report_unchosen_conversion_cutoffs(pb_circuit)
    pb_circuit_bytes = pb_circuit.SerializeToString()

    started_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    with _job_failure_reported_uniformly():
        pb_result_bytes = simulate_circuit(
            pb_circuit_bytes,
            backend,
            n_shots,
            state_save_policy,
            seed=seed,
            max_bond_dim=max_bond_dim,
            svd_cutoff=svd_cutoff,
            track_truncation_error=track_truncation_error,
            max_saved_state_elements=max_saved_state_elements,
            # The caller's option, unchanged: which parts of it this simulator implements is the
            # simulator's business, not something to be second-guessed from here.
            state_representation=state_representation,
        )
    pb_result = QuantumProgramResult()
    pb_result.ParseFromString(pb_result_bytes)

    # A run is exclusively bosonic or Fock, so only one of these proto fields is populated. Pure and
    # mixed Fock states share circuit_fock_quantum_state, so shot order is preserved either way.
    saved_protos: list = list(pb_result.circuit_state) + list(pb_result.circuit_fock_quantum_state)
    states = _states_by_shot(saved_protos, state_save_policy)
    truncation_errors = list(pb_result.circuit_truncation_error)
    circuit_result = CircuitResult.construct_from_proto(proto_result=pb_result.circuit_result)
    # There is one entry per requested shot, which is what was asked for, rather than per entry of any
    # one result field: a shot that measured nothing still ran and still has a state. The measured
    # values are shared with circuit_result rather than copied, both describing the same shot.
    measured_values = list(circuit_result)
    shot_results = [
        ShotResult(
            measured_value=measured_values[shot] if shot < len(measured_values) else CircuitShotMeasuredValue(),
            state=states.get(shot),
            truncation_error=truncation_errors[shot] if shot < len(truncation_errors) else None,
        )
        for shot in range(n_shots)
    ]
    finished_at = datetime.now(started_at.tzinfo)

    return LocalCircuitResult(
        execution_time=finished_at - started_at,
        circuit_result=circuit_result,
        shot_results=shot_results,
    )


@dataclass(frozen=True)
class LocalGraphResult:
    """The result of executing a quantum circuit with the Claveles graph simulator."""

    execution_time: timedelta
    """The time to execute a quantum circuit."""

    graph_result: GraphResult
    """Measurement results after circuit execution."""


def local_run_graph(
    backend: Backend, n_shots: int, squeezing_level: float, graph: GraphRepr, seed: int | None = None
) -> LocalGraphResult:
    pb_graph = graph.proto()
    pb_graph_bytes = pb_graph.SerializeToString()

    started_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    with _job_failure_reported_uniformly():
        pb_result_bytes = simulate_graph(pb_graph_bytes, backend, n_shots, squeezing_level, seed=seed)
    pb_result = PbGraphResult()
    pb_result.ParseFromString(pb_result_bytes)
    finished_at = datetime.now(started_at.tzinfo)

    return LocalGraphResult(
        execution_time=finished_at - started_at,
        graph_result=GraphResult.construct_from_proto(proto_result=pb_result),
    )


@dataclass(frozen=True)
class LocalMachineryResult:
    """The result of executing a quantum circuit with the Claveles machinery simulator."""

    execution_time: timedelta
    """The time to execute a quantum circuit."""

    machinery_result: MachineryResult
    """Measurement results after circuit execution."""


def local_run_machinery(
    backend: Backend, n_shots: int, squeezing_level: float, machinery: MachineryRepr, seed: int | None = None
) -> LocalMachineryResult:
    pb_machinery = machinery.proto()
    pb_machinery_bytes = pb_machinery.SerializeToString()

    started_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    with _job_failure_reported_uniformly():
        pb_result_bytes = simulate_machinery(pb_machinery_bytes, backend, n_shots, squeezing_level, seed=seed)
    pb_result = PbMachineryResult()
    pb_result.ParseFromString(pb_result_bytes)
    finished_at = datetime.now(started_at.tzinfo)

    return LocalMachineryResult(
        execution_time=finished_at - started_at,
        machinery_result=MachineryResult.construct_from_proto(proto_result=pb_result),
    )
