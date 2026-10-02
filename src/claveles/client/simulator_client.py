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
#
# Portions of this file are derived from MQC3,
# originally licensed under the MIT License.

"""Simulator client for optical quantum computing."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal, SupportsIndex

from claveles.circuit import CircuitRepr, CircuitResult
from claveles.client.abstract import (
    AbstractClient,
    AbstractClientResult,
    MeasuredValue,
    ReprType,
    ResultType,
    check_representation,
    representation_kind,
    result_kind,
    unsupported_representation_message,
)
from claveles.graph import GraphRepr, GraphResult
from claveles.machinery import MachineryRepr, MachineryResult
from claveles.warnings import IgnoredOptionWarning

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from zoneinfo import ZoneInfo

    from claveles.circuit.result import CircuitShotMeasuredValue
    from claveles.circuit.state import QuantumState
    from claveles.support import Representation, StateRepresentationOption


_TZ_UTC = timezone.utc  # noqa: UP017


@dataclass
class JobTimeline:
    """Timeline of a job execution in UTC."""

    submitted_at: datetime | None = None
    queued_at: datetime | None = None
    dequeued_at: datetime | None = None
    execution_started_at: datetime | None = None
    execution_finished_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def wait_time(self) -> timedelta | None:
        """Return the waiting time of the job.

        ``wait_time`` is the time between the job being queued and the job being dequeued.

        Returns:
            timedelta | None: Waiting time of the job.
        """
        if self.queued_at is None or self.dequeued_at is None:
            return None
        return self.dequeued_at - self.queued_at

    @property
    def execution_time(self) -> timedelta | None:
        """Return the execution time of the job.

        Returns:
            timedelta | None: Execution time of the job.
        """
        if self.execution_started_at is None or self.execution_finished_at is None:
            return None
        return self.execution_finished_at - self.execution_started_at

    @property
    def total_time(self) -> timedelta | None:
        """Return the total time of the job.

        ``total_time`` is the time between the job being submitted and the job being finished.

        Returns:
            timedelta | None: Total time of the job.
        """
        if self.submitted_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.submitted_at


@dataclass
class ExecutionDetails:
    """Versions and timestamps related to a job execution."""

    scheduler_version: str
    simulator_version: str
    timeline: JobTimeline | None = None


@dataclass(frozen=True)
class ShotResult:
    """What one shot of a circuit simulation produced.

    A shot measures some modes and leaves the rest carrying a state, so those two halves are reported
    together rather than as separate lists that the caller has to line up. Every mode is measured at
    most once, so :attr:`measured_value` has one entry per measured mode.
    """

    measured_value: CircuitShotMeasuredValue
    """The values this shot measured, one per measured mode."""

    state: QuantumState | None
    """The modes this shot did not measure, or None when the state of this shot was not saved.

    A shot that measured every mode leaves a state with no modes, which is distinct from the state not
    having been saved. Which shots are saved is set by
    :attr:`SimulatorClient.state_save_policy`, and a state too large to hand back is refused rather than
    dropped, so this is None exactly when the policy did not ask for this shot.

    The state holds the surviving modes as a single component, a
    :class:`~claveles.circuit.state.BosonicState` for bosonic circuits or a
    :class:`~claveles.circuit.state.FockState` for Fock circuits; a single run is exclusively one or the
    other. For a :class:`~claveles.circuit.state.FockState`, use ``is_pure`` to tell whether it carries a
    state vector or a density matrix.
    """

    truncation_error: float | None
    """Accumulated SVD truncation error of this shot, or None when it was not tracked.

    Populated only when ``track_truncation_error`` is enabled on the client.
    """


@dataclass(frozen=True)
class SimulatorClientResult(AbstractClientResult):
    """The result of executing a representation with a SimulatorClient."""

    execution_details: ExecutionDetails
    """Execution details includes version of backend and the timeline of the job."""

    _result: ResultType
    """What the run produced: the one result the representation it ran produces.

    A single field rather than one ``Optional`` per representation. The three could only ever describe
    one run between them, so two of them were always ``None`` and a reader had to know that; as three
    fields they could also describe states no run produces -- none set, or two at once -- which
    :attr:`execution_result` then had to have an error path for. ``circuit_result``,
    ``graph_result`` and ``machinery_result`` still read it back per representation, and now say what
    ran when asked for a representation that did not.
    """

    _shot_results: list[ShotResult] | None
    """What each shot produced, or None if the execution was not a circuit.

    None for graph and machinery in every mode. An empty list would say the run had no shots, which is
    a different answer, and a caller writing ``if result.shot_results:`` would then be reading a fact
    about the representation as a fact about the data.
    """

    @property
    def shot_results(self) -> list[ShotResult]:
        """Return what each shot produced, in shot order.

        Returns:
            list[ShotResult]: One entry per shot, pairing the values it measured with the state it
                left behind.

        Raises:
            ValueError: If the execution was not a circuit. Graph and machinery execution report
                measured values only, so there is nothing to pair. Both ``mode="local"`` and
                ``mode="simulator-server"`` refuse it this way, rather than one of them returning an
                empty list, so the same graph run does not raise in one mode and not in the other.
        """
        if self._shot_results is None:
            msg = (
                "Shot results are only available for circuit execution. Run a circuit first, or read "
                "the measured values from graph_result or machinery_result."
            )
            raise ValueError(msg)
        return self._shot_results

    def _result_of_kind_message(self, asked_for: Representation) -> str:
        """Describe a request for a result of a representation this run was not.

        Args:
            asked_for: The representation whose result was asked for.

        Returns:
            str: The message to raise a ``ValueError`` with, naming what ran instead. A run produces
                the one result of the representation it was given, so naming that representation is
                what tells a caller which accessor to read. A value error rather than a type error:
                nothing was passed in to be of the wrong type, the caller asked this result for a
                level it does not hold -- which is what the accessors' ``noqa: TRY004`` records.
        """
        ran = result_kind(self._result)
        return (
            f"{asked_for.capitalize()} result is not available: this run was a {ran}, so it produced a "
            f"{ran} result. Read it from {ran}_result, or run a {asked_for} to get a {asked_for} result."
        )

    @property
    def circuit_result(self) -> CircuitResult:
        """Return the circuit execution result.

        Returns:
            CircuitResult: The circuit execution result.

        Raises:
            ValueError: If the run was a graph or a machinery program, which produce no circuit-level
                result. The message names what ran.
        """
        if not isinstance(self._result, CircuitResult):
            raise ValueError(self._result_of_kind_message("circuit"))  # noqa: TRY004
        return self._result

    @property
    def graph_result(self) -> GraphResult:
        """Return the graph execution result.

        Returns:
            GraphResult: The graph execution result.

        Raises:
            ValueError: If the run was a circuit or a machinery program, which produce no graph-level
                result. The message names what ran.
        """
        if not isinstance(self._result, GraphResult):
            raise ValueError(self._result_of_kind_message("graph"))  # noqa: TRY004
        return self._result

    @property
    def machinery_result(self) -> MachineryResult:
        """Return the machinery execution result.

        Returns:
            MachineryResult: The machinery execution result.

        Raises:
            ValueError: If the run was a circuit or a graph, which produce no machinery-level result.
                The message names what ran.
        """
        if not isinstance(self._result, MachineryResult):
            raise ValueError(self._result_of_kind_message("machinery"))  # noqa: TRY004
        return self._result

    @property
    def execution_result(self) -> ResultType:
        """Get the result of the representation that was run.

        Always available: a run produces exactly one result, and this returns it.

        Returns:
            ResultType: The result of the representation that was run.

        Examples:
            >>> from math import pi, sqrt
            >>> from claveles.client import SimulatorClient
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> from claveles.circuit.state import BosonicState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | BosonicState.vacuum()
            [QuMode(id=0)]
            >>> circuit.Q(0) | std.Displacement((0.1 + 0.2j) / sqrt(2))
            [QuMode(id=0)]
            >>> circuit.Q(0)| std.MeasureHomodyne(pi / 2)
            Var([std.measure_homodyne] [1.5707963267948966] [0])
            >>> client = SimulatorClient(n_shots=1, backend="cpu")
            >>> result = client.run(circuit)
            >>> circuit_result = result.execution_result
        """
        return self._result

    @property
    def wait_time(self) -> timedelta | None:
        """Return the waiting time of the job.

        ``wait_time`` is the time between the job being queued and the job being dequeued.

        Returns:
            timedelta | None: The waiting time of the job.
        """
        if self.execution_details.timeline is None:
            return None

        return self.execution_details.timeline.wait_time

    @property
    def execution_time(self) -> timedelta | None:
        """Return the execution time of the job.

        Returns:
            timedelta | None: The execution time of the job.
        """
        if self.execution_details.timeline is None:
            return None

        return self.execution_details.timeline.execution_time

    @property
    def total_time(self) -> timedelta | None:
        """Return the total time of the job.

        ``total_time`` is the time between the job being submitted and the job being finished.

        Returns:
            timedelta | None: The total time of the job.
        """
        if self.execution_details.timeline is None:
            return None

        return self.execution_details.timeline.total_time

    def __len__(self) -> int:
        """Return the number of shots.

        Returns:
            int: The number of shots.
        """
        return self.n_shots

    def __iter__(self) -> Iterator:
        """Iterator of the result.

        The iterator is the same as that of ``self.execution_result``.

        Yields:
            Iterator: Iterator of the result.
        """
        yield from self.execution_result

    def get_shot_measured_value(self, index: int) -> MeasuredValue:
        """Get the measured value of the shot at the index.

        This function gets values from ``self.execution_result``.

        Args:
            index (int): Shot index.

        Returns:
            MeasuredValue: Measured value.
        """
        return self.execution_result.get_shot_measured_value(index)

    def __getitem__(
        self,
        index: int | slice | SupportsIndex,
    ) -> MeasuredValue | Sequence[MeasuredValue]:
        """Get the measured value of the shot at the index.

        This function gets values from ``self.execution_result``.

        Args:
            index (int | slice | SupportsIndex): Index or slice.

        Returns:
            MeasuredValue | Sequence[MeasuredValue]: Measured value or sequence of measured values.
        """
        return self.execution_result.measured_vals[index]


BackendType = Literal["cpu", "single_cpu", "multi_cpu", "multi_cpu_peak", "multi_cpu_shot", "single_gpu", "multi_gpu"]
"""Backends for the simulator client.

- ``cpu``: Alias for ``multi_cpu`` (auto-selecting multi-threaded CPU execution). Fock circuits
  always run on a single thread regardless of this choice.
- ``single_cpu``: Simulate circuits with CPU on a single thread.
- ``multi_cpu``: Auto-selecting multi-threaded CPU execution, which picks between the two strategies
  below.
- ``multi_cpu_peak`` / ``multi_cpu_shot``: The peak-level and shot-level multi-threaded CPU
  strategies, asked for by name. Distinct backends in both modes -- the simulator maps each to its own
  backend rather than folding them into ``multi_cpu``.
- ``single_gpu``: Simulate circuits with a single GPU. Fock circuits are not supported.
- ``multi_gpu``: Simulate circuits with multi GPU. Fock circuits are not supported.
"""

SimulatorMode = Literal["local", "simulator-server", "remote"]
"""Where a :class:`SimulatorClient` executes a representation.

- ``local``: Run with the ``claveles_simulator`` wheel installed in this process.
- ``simulator-server``: Submit directly to a running Cloud Simulator Server (CSS) instance over
  HTTP. This is a verification/development path, not the production path -- the production path is
  ``mode="remote"``.
- ``remote``: Submit through the Cloud User API (the same endpoint and Bearer token
  :class:`~claveles.client.OptQCClient` uses with ``target="cloud"``), which relays the job to CSS through
  the ``simulator-bridge`` service. This is the production remote path.
"""

StateSavePolicy = Literal["all", "first_only", "none"]
"""Policy for saving states after simulation.

- ``all``: Save states of all shots.
- ``first_only``: Save only the state of the first shot.
- ``none``: Do not save any states.
"""

SIMULATOR_SERVER_STATE_SAVE_POLICY: StateSavePolicy = "first_only"
"""The save policy the Cloud Simulator Server applies, which a client cannot change.

Named rather than repeated as a literal, so that the value
:attr:`SimulatorClient.state_save_policy` reports for ``mode="simulator-server"`` and the value the
result decoder is told to expect are the same one.
"""

DEFAULT_MAX_SAVED_STATE_ELEMENTS = 1 << 20
"""Default upper bound on the number of elements in a single saved Fock state.

Matches the simulator's own default. See
:attr:`SimulatorClient.max_saved_state_elements`.
"""

DEFAULT_RESOURCE_SQUEEZING_LEVEL = 10.0
"""Squeezing level of the resource state used when the caller does not choose one.

See :attr:`SimulatorClient.resource_squeezing_level`.
"""


class _Unspecified:
    """Stands for an option the caller did not pass at all.

    Telling "not given" apart from "given the value that happens to be the default" is what lets a mode
    that cannot honour an option refuse it outright. Refusing only the values that *differ* from the
    default accepts the default and ignores it, and refuses ``None`` -- the documented way to lift the
    saved-state bound -- which is the wrong way round.
    """

    def __repr__(self) -> str:
        """Return a form that reads as the absence it stands for.

        Returns:
            str: ``"<unspecified>"``.
        """
        return "<unspecified>"


_UNSPECIFIED = _Unspecified()

_CIRCUIT_ONLY_OPTIONS: dict[str, object] = {
    "max_bond_dim": None,
    "svd_cutoff": 0.0,
    "track_truncation_error": False,
    "state_representation": "auto",
    "decompose_std": False,
}
"""Options only a circuit run reads, each mapped to the value that asks for nothing.

The value is the option's inert setting rather than its default: ``svd_cutoff=0.0`` disables
threshold-based truncation and ``state_representation="auto"`` leaves the choice to the circuit, so
passing either on a graph run asks for nothing and gets it. Anything else was asked for and dropped,
which is what :class:`~claveles.warnings.IgnoredOptionWarning` reports.
"""

_STATE_SAVING_OPTIONS_INERT_WHEN_LOCAL: dict[str, object] = {
    "state_save_policy": "none",
    "max_saved_state_elements": DEFAULT_MAX_SAVED_STATE_ELEMENTS,
}
"""The circuit-only options that govern state saving, each mapped to its inert value under ``local``.

Kept out of :data:`_CIRCUIT_ONLY_OPTIONS` because their inert value depends on the mode, which a flat
table cannot express. Only ``mode="local"`` lets a caller choose either one; the other modes write the
Cloud Simulator Server's own policy and bound into them, and a value nobody chose is not a value to
report as dropped. So these count as asked for under ``local`` only.
"""

_GRAPH_AND_MACHINERY_ONLY_OPTIONS: dict[str, object] = {
    "resource_squeezing_level": None,
}
"""Options only a graph or machinery run reads, each mapped to the value that asks for nothing.

A squeezing level has no inert value -- every number is a real level -- so ``None`` stands for
"unchosen" here, and :data:`DEFAULT_RESOURCE_SQUEEZING_LEVEL` is what an unchosen one resolves to.
"""


def _resolve_server_url_and_api_key(mode: SimulatorMode, url: str | None, api_key: str | None) -> tuple[str, str]:
    """Resolve the URL/API key pair for ``mode="simulator-server"`` or ``mode="remote"``.

    Pulled out of the constructor because the two modes read a different environment variable pair,
    and each has its own message when neither an explicit value nor its environment variables supply
    one -- inlining both here kept ``__init__`` a single, undifferentiated block of branches.

    Args:
        mode: Either ``"simulator-server"`` or ``"remote"``.
        url: The caller's explicit ``url``, or ``None``.
        api_key: The caller's explicit ``api_key``, or ``None``.

    Returns:
        The resolved ``(url, api_key)`` pair.

    Raises:
        ValueError: If neither an explicit value nor the mode's environment variables supply both.
    """
    if mode == "simulator-server":
        resolved_url = url or os.getenv("OPTQC_SIMULATOR_URL")
        resolved_api_key = api_key or os.getenv("OPTQC_SIMULATOR_API_KEY")
        if not resolved_url or not resolved_api_key:
            msg = (
                "mode='simulator-server' requires url and api_key, either passed directly or "
                "via the OPTQC_SIMULATOR_URL / OPTQC_SIMULATOR_API_KEY environment variables."
            )
            raise ValueError(msg)
        return resolved_url, resolved_api_key

    # mode == "remote": the same Cloud User API endpoint and Bearer token as
    # OptQCClient(target="cloud"), not the OPTQC_SIMULATOR_URL / OPTQC_SIMULATOR_API_KEY pair above,
    # which name a CSS instance directly and skip the Cloud User API and simulator-bridge entirely.
    resolved_url = url or os.getenv("OPTQC_CLIENT_URL")
    resolved_api_key = api_key or os.getenv("OPTQC_CLIENT_API_KEY")
    if not resolved_url or not resolved_api_key:
        msg = (
            "mode='remote' requires url and api_key -- the same Cloud User API endpoint and Bearer "
            "token used with OptQCClient(target='cloud') -- either passed directly or via the "
            "OPTQC_CLIENT_URL / OPTQC_CLIENT_API_KEY environment variables."
        )
        raise ValueError(msg)
    return resolved_url, resolved_api_key


class SimulatorClient(AbstractClient):
    """Simulator client for optical quantum computing."""

    mode: SimulatorMode
    """Where this client executes a representation.

    See :data:`SimulatorMode`.
    """

    backend: BackendType | None
    """Backend type of the simulator.

    ``None`` means "unspecified": for ``mode="local"`` this behaves like ``"cpu"`` (today's
    default); for ``mode="simulator-server"`` and ``mode="remote"`` the backend is omitted from the
    request and the Cloud Simulator Server resolves it automatically (Bosonic -> ``single_gpu``,
    Fock -> ``single_cpu``).

    See :data:`BackendType`.
    """

    state_save_policy: StateSavePolicy
    """Policy the run will actually apply when saving states.

    Always a policy rather than ``None``, so that reading it answers "will my shots carry a state?"
    without knowing which mode resolves an unspecified policy to what. An unspecified policy is
    ``"none"`` for ``mode="local"``, and ``"first_only"`` for ``mode="simulator-server"`` and
    ``mode="remote"``, which is what the Cloud Simulator Server applies and does not let a client
    change. The modes therefore genuinely differ here, and this attribute is where the difference is
    visible rather than surprising.

    Assigning to it is honoured for ``mode="local"``. For ``mode="simulator-server"`` and
    ``mode="remote"`` it is checked again at :meth:`run` and anything other than ``"first_only"`` is
    refused, rather than accepted and ignored on the way out.

    See :data:`StateSavePolicy`.
    """

    state_representation: StateRepresentationOption
    """Which representation the simulator holds the state in.

    Unlike :attr:`backend`, which selects the hardware the simulation runs on, this selects how the
    state itself is held: as a weighted sum of Gaussian components, or in the truncated Fock basis.

    ``"auto"`` (the default) runs in the bosonic representation and converts to the Fock
    representation at the first operation only that representation can apply, which is what lets a
    circuit apply a non-Gaussian gate to a Gaussian state at all. Until the simulator can convert
    partway through, a circuit that has to reach the Fock representation is converted up front
    instead, which runs the whole circuit there: the operations before the one that forced it are then
    applied in the truncated Fock basis rather than exactly, so expect results on such circuits to
    shift slightly once converting partway through arrives. ``"bosonic"`` refuses such a circuit
    instead of converting, which is the way to be sure a run stays Gaussian. ``"fock"`` converts every
    bosonic state up front and runs the whole circuit in the Fock basis, which is how the same circuit
    can be compared against its exact bosonic result.

    A conversion truncates each mode at its state's
    :attr:`~claveles.circuit.state.BosonicState.cutoff_dims`; where none was chosen the simulator picks a
    dimension for it, since no single value suits every state. Only the single-threaded CPU backend can
    run a Fock state, so a converting circuit cannot use a GPU backend.

    ``mode="remote"`` accepts only ``"auto"``: the Cloud User API's job settings have no field to carry
    an explicit choice through to CSS, so :meth:`run` refuses any other value instead of letting CSS's
    own auto-resolution silently override a choice it was never told about. Use
    ``mode="simulator-server"`` to choose ``"bosonic"`` or ``"fock"`` against a CSS instance.

    See :data:`~claveles.support.StateRepresentationOption`.
    """

    resource_squeezing_level: float | None
    """Squeezing level of the resource state, or ``None`` to leave it unchosen.

    An unchosen level resolves to :data:`DEFAULT_RESOURCE_SQUEEZING_LEVEL` at run time. ``None`` is the
    default rather than that number so that "I did not choose" stays distinguishable from "I chose 10.0"
    -- a graph or machinery program is what reads this, and a circuit run that was given a level reports
    an :class:`~claveles.warnings.IgnoredOptionWarning` rather than dropping it in silence.
    """

    max_bond_dim: int | None
    """Maximum bond dimension of the Fock simulator's matrix product state.

    ``None`` leaves the bond dimension unbounded, so the simulation is exact. Used only for Fock
    circuit simulations. Raising this is recommended for circuits that hold a mixed state, whose
    purification needs a larger bond dimension than an equivalent unitary circuit.
    """

    svd_cutoff: float
    """SVD truncation cutoff of the Fock simulator.

    Singular values below this threshold are discarded. ``0.0`` disables threshold-based truncation,
    so the simulation is exact. Used only for Fock circuit simulations.
    """

    track_truncation_error: bool
    """Whether to record the accumulated SVD truncation error of each shot.

    When True, the per-shot error is exposed via :attr:`ShotResult.truncation_error`.
    Used only for Fock circuit simulations.
    """

    max_saved_state_elements: int | None
    """Bound on the number of elements in a single saved Fock state that this client asks for.

    A saved state is handed back as a dense array whose size grows exponentially with the number of
    modes, so saving one beyond this bound fails rather than exhausting memory. A mixed state
    (density matrix) reaches the bound at fewer modes than a pure one (state vector), since it holds
    the square of the state vector's elements.

    ``None`` means this client asks for no bound of its own. For ``mode="local"`` that leaves the
    simulation unbounded; for ``mode="simulator-server"`` and ``mode="remote"`` it is the only
    possible value, the Cloud Simulator Server applying a bound of its own that it does not report.
    Defaults to :data:`DEFAULT_MAX_SAVED_STATE_ELEMENTS` for ``mode="local"``. Read only for Fock
    circuit simulations, and only where :attr:`state_save_policy` saves anything at all.
    """

    timezone: timezone | ZoneInfo
    """Timezone to use."""

    url: str | None
    """Base URL of the server this client talks to. For ``mode="simulator-server"``, a Cloud
    Simulator Server instance. For ``mode="remote"``, the Cloud User API (the same one
    :class:`~claveles.client.OptQCClient` uses with ``target="cloud"``, e.g.
    ``https://compute.cloud.optqc.com``). Unset for ``mode="local"``."""

    api_key: str | None
    """Bearer token accepted by the server named in :attr:`url`. For ``mode="simulator-server"``, a
    token specific to that Cloud Simulator Server instance. For ``mode="remote"``, the same user API
    key used with :class:`~claveles.client.OptQCClient`. Unset for ``mode="local"``."""

    timeout_msec: int | None
    """Server-side execution timeout in milliseconds. Only used for ``mode="simulator-server"`` and
    ``mode="remote"``."""

    polling_interval: float
    """Interval in seconds between job status polls. Only used for ``mode="simulator-server"`` and
    ``mode="remote"``."""

    def __init__(  # noqa: PLR0913  # public constructor exposing simulator configuration options
        self,
        n_shots: int = 1024,
        backend: BackendType | None = None,
        state_save_policy: StateSavePolicy | None = None,
        resource_squeezing_level: float | None = None,
        random_seed: int | None = None,
        *,
        mode: SimulatorMode = "local",
        state_representation: StateRepresentationOption = "auto",
        url: str | None = None,
        api_key: str | None = None,
        timeout_msec: int | None = None,
        polling_interval: float = 1.0,
        max_bond_dim: int | None = None,
        svd_cutoff: float = 0.0,
        track_truncation_error: bool = False,
        max_saved_state_elements: int | _Unspecified | None = _UNSPECIFIED,
        decompose_std: bool = False,
        timezone: timezone | ZoneInfo = _TZ_UTC,
    ) -> None:
        """SimulatorClient constructor.

        Args:
            n_shots (int): The number of shots.
            backend (:data:`BackendType` | None): Backend type. ``None`` (default) behaves like
                ``"cpu"`` for ``mode="local"``; for ``mode="simulator-server"`` and ``mode="remote"``
                it is omitted from the request so the server resolves it automatically.
            state_save_policy (:data:`StateSavePolicy` | None): Policy for saving states after
                simulation (all, first_only or none). ``None`` (the default) leaves it unchosen, which
                resolves to ``"none"`` for ``mode="local"`` and to ``"first_only"`` for
                ``mode="simulator-server"``/``mode="remote"``. Only ``None`` is accepted for those
                modes, the Cloud Simulator Server always applying ``first_only`` and not honouring a
                client-chosen policy. :attr:`state_save_policy` reports whichever policy the run will
                apply.
            resource_squeezing_level (float | None): Squeezing level of the resource state, read by a
                graph or machinery run. ``None`` (the default) leaves it unchosen and resolves to
                :data:`DEFAULT_RESOURCE_SQUEEZING_LEVEL`; giving a level for a circuit run reports an
                :class:`~claveles.warnings.IgnoredOptionWarning`.
            random_seed (int | None): Optional random seed. If None, a fresh random seed is chosen
                for each run, so separate seedless runs sample independently; pass an explicit seed
                to reproduce a run deterministically. This holds in every mode. For ``mode="local"``
                the randomly chosen seed is not returned, so pass an explicit seed if you need to
                reproduce a local run. For ``mode="remote"`` (the production path through the Cloud
                User API) the seed is randomized on the server, but the relayed
                ``SimulatorJobExecutionReport`` carries no ``effective_seed``, so the chosen value is
                likewise not returned to the client -- pass an explicit seed to reproduce a remote
                run. Within a single run, each shot draws from its own random stream derived from the
                seed and the shot index.
            mode (:data:`SimulatorMode`): Where to execute (default: ``"local"``). See
                :data:`SimulatorMode`.
            state_representation (:data:`~claveles.support.StateRepresentationOption`): How the simulator
                holds the state -- ``"bosonic"``, ``"fock"``, or ``"auto"`` (the default), which
                converts to the Fock representation where the circuit requires it. Used only for
                circuit simulations. ``mode="remote"`` accepts only ``"auto"``. See
                :attr:`state_representation`.
            url (str | None): Base URL of the server to talk to. Used only for
                ``mode="simulator-server"`` (a Cloud Simulator Server instance; if not specified,
                retrieved from the environment variable ``"OPTQC_SIMULATOR_URL"``) and
                ``mode="remote"`` (the Cloud User API; if not specified, retrieved from
                ``"OPTQC_CLIENT_URL"``, the same environment variable
                :class:`~claveles.client.optqc_client.OptQCClient` reads).
            api_key (str | None): Bearer token accepted by the server named in ``url``. Used only for
                ``mode="simulator-server"`` (a token specific to that server instance, not the API key
                used with :class:`~claveles.client.optqc_client.OptQCClient`; if not specified, retrieved
                from ``"OPTQC_SIMULATOR_API_KEY"``) and ``mode="remote"`` (the same user API key used
                with :class:`~claveles.client.optqc_client.OptQCClient`; if not specified, retrieved from
                ``"OPTQC_CLIENT_API_KEY"``).
            timeout_msec (int | None): Server-side execution timeout in milliseconds. Used only for
                ``mode="simulator-server"`` and ``mode="remote"``. ``None`` lets the server apply its
                own default.
            polling_interval (float): Interval in seconds between job status polls. Used only for
                ``mode="simulator-server"`` and ``mode="remote"``.
            max_bond_dim (int | None): Maximum MPS bond dimension for the Fock simulator. None leaves the
                bond dimension unbounded (exact simulation). Used only for Fock circuit simulations.
            svd_cutoff (float): SVD truncation cutoff for the Fock simulator. 0.0 disables threshold-based
                truncation (exact simulation). Used only for Fock circuit simulations.
            track_truncation_error (bool): When True, the per-shot accumulated SVD truncation error is
                recorded and exposed via ``ShotResult.truncation_error``. Used only for Fock
                circuit simulations.
            max_saved_state_elements (int | None): Maximum number of elements in a single saved Fock
                state. Saving a larger one raises an error suggesting a partial trace, instead of
                building a dense array that grows exponentially with the number of modes. ``None``
                asks for no bound. Left unspecified it defaults to
                :data:`DEFAULT_MAX_SAVED_STATE_ELEMENTS` for ``mode="local"``; giving it any value at
                all, ``None`` included, is refused for ``mode="simulator-server"`` and
                ``mode="remote"``, which apply a bound of the server's own.
            decompose_std (bool): When True, std operations are lowered to their intrinsic decomposition
                before the circuit is simulated (e.g. ``std.BeamSplitter`` becomes phase rotations plus
                ``intrinsic.Manual``). When False (the default), std operations that have a dedicated proto
                op type are simulated natively (e.g. ``std.BeamSplitter`` via ``BeamSplitterStd``). Used
                only for circuit simulations.
            timezone (timezone | ZoneInfo): Timezone to use (default: UTC).

        Raises:
            ValueError: If an argument the chosen mode cannot honour is given, if
                ``mode="simulator-server"``/``mode="remote"`` is missing a URL/API key, or if
                ``state_representation`` is not one of the three accepted values.

        Examples:
            >>> from claveles.client import SimulatorClient
            >>> client = SimulatorClient(
            ...     n_shots=1024,
            ...     backend="cpu",
            ...     state_save_policy="first_only",
            ... )
        """
        AbstractClient.__init__(self, n_shots=n_shots, tz=timezone)
        self.mode = mode
        self.backend = backend
        self.state_representation = state_representation
        self.resource_squeezing_level = resource_squeezing_level
        self.random_seed = random_seed
        self.max_bond_dim = max_bond_dim
        self.svd_cutoff = svd_cutoff
        self.track_truncation_error = track_truncation_error
        self.decompose_std = decompose_std
        self.timeout_msec = timeout_msec
        self.polling_interval = polling_interval

        # Checked rather than left to the type annotation, because the requirements document names
        # these options in capitalized form and a caller following it would otherwise have the value
        # silently treated as an unknown representation further down.
        if state_representation not in {"bosonic", "fock", "auto"}:
            msg = (
                f"Unknown state_representation: {state_representation!r}. Expected 'bosonic', 'fock' or 'auto', "
                "in lower case."
            )
            raise ValueError(msg)

        # Each mode resolves the two state-saving options to what it can actually deliver, so that the
        # attributes report the run's behaviour instead of the caller's request. The server modes
        # refuse any value they were given for them rather than only the values that differ from a
        # default: being handed the default and ignoring it, while refusing None, told a caller the
        # opposite of the truth about which of the two it may pass.
        if mode == "local":
            if url is not None or api_key is not None:
                msg = "url and api_key are only used for mode='simulator-server'/'remote', not mode='local'."
                raise ValueError(msg)
            self.url = None
            self.api_key = None
            self.state_save_policy = state_save_policy or "none"
            self.max_saved_state_elements = (
                DEFAULT_MAX_SAVED_STATE_ELEMENTS
                if isinstance(max_saved_state_elements, _Unspecified)
                else max_saved_state_elements
            )
        elif mode in {"simulator-server", "remote"}:
            if state_save_policy is not None:
                msg = (
                    f"state_save_policy cannot be set for mode={mode!r}: the Cloud Simulator Server "
                    "always applies 'first_only' and does not accept a client-chosen policy. Leave it "
                    "unset and read it back to see that."
                )
                raise ValueError(msg)
            if not isinstance(max_saved_state_elements, _Unspecified):
                msg = (
                    f"max_saved_state_elements cannot be set for mode={mode!r}: the Cloud Simulator "
                    "Server applies a bound of its own, which it does not report. Leave it unset."
                )
                raise ValueError(msg)
            if mode == "remote" and state_representation != "auto":
                msg = (
                    f"state_representation cannot be set to {state_representation!r} for mode='remote': "
                    "the Cloud User API's job settings have no field to carry an explicit choice "
                    "through to the Cloud Simulator Server, so only the default 'auto' is accepted. "
                    "Use mode='simulator-server' to choose 'bosonic' or 'fock' against a CSS instance "
                    "directly."
                )
                raise ValueError(msg)
            self.state_save_policy = SIMULATOR_SERVER_STATE_SAVE_POLICY
            self.max_saved_state_elements = None
            self.url, self.api_key = _resolve_server_url_and_api_key(mode, url, api_key)
        else:
            msg = f"Unknown mode: {mode!r}. Expected 'local', 'simulator-server' or 'remote'."
            raise ValueError(msg)

    def _squeezing_level(self) -> float:
        """Return the resource squeezing level to run with.

        Returns:
            float: The level the caller chose, or :data:`DEFAULT_RESOURCE_SQUEEZING_LEVEL` where they
                left it unchosen.
        """
        if self.resource_squeezing_level is None:
            return DEFAULT_RESOURCE_SQUEEZING_LEVEL
        return self.resource_squeezing_level

    def _check_mode_can_still_honour_its_options(self) -> None:
        """Refuse an option the mode cannot deliver, checking the attributes rather than the arguments.

        The options are public and documented as assignable, so the constructor's checks only cover the
        value they were built with. Without this, ``mode="simulator-server"`` would accept a policy
        assigned afterwards and then ignore it, its path having ``first_only`` written into it directly.

        Raises:
            ValueError: If an attribute now holds something the mode cannot honour.
        """
        if self.mode not in {"simulator-server", "remote"}:
            return
        if self.state_save_policy != SIMULATOR_SERVER_STATE_SAVE_POLICY:
            msg = (
                f"state_save_policy is {self.state_save_policy!r}, but mode={self.mode!r} always "
                f"applies {SIMULATOR_SERVER_STATE_SAVE_POLICY!r} and cannot honour another policy. "
                "Run with mode='local' to choose one."
            )
            raise ValueError(msg)
        if self.max_saved_state_elements is not None:
            msg = (
                f"max_saved_state_elements is {self.max_saved_state_elements!r}, but "
                f"mode={self.mode!r} applies a bound of the server's own and cannot honour one set "
                "here. Run with mode='local' to choose one."
            )
            raise ValueError(msg)

    def _circuit_only_options_asked_for(self) -> list[str]:
        """Name the circuit-only options this client was actually asked to apply.

        Every circuit-only option has to be listed here, not just the ones with a mode-independent
        inert value: an option a graph run drops without saying so is read back as a fact about the
        run rather than as an option that went nowhere, which is what this warning exists to prevent.

        Returns:
            list[str]: The option names, sorted, so that a message listing them reads the same way
                whatever order the tables happen to be in.
        """
        asked = [name for name, inert in _CIRCUIT_ONLY_OPTIONS.items() if getattr(self, name) != inert]
        if self.mode == "local":
            asked += [
                name for name, inert in _STATE_SAVING_OPTIONS_INERT_WHEN_LOCAL.items() if getattr(self, name) != inert
            ]
        return sorted(asked)

    def _warn_about_options_this_run_will_not_read(self, representation: ReprType) -> None:
        """Report options that were set but do not apply to the representation being run.

        The client takes every option whatever it is asked to run and drops the ones the run does not
        read. Dropping them silently is what let ``track_truncation_error=True`` on a graph run return
        ``truncation_error=None`` and read as "nothing was truncated".

        Args:
            representation: What is about to run.
        """
        kind = representation_kind(representation)
        is_circuit = kind == "circuit"
        if is_circuit:
            ignored = sorted(
                name for name, inert in _GRAPH_AND_MACHINERY_ONLY_OPTIONS.items() if getattr(self, name) != inert
            )
        else:
            ignored = self._circuit_only_options_asked_for()
        if not ignored:
            return

        ran = f"a {kind}"
        reads = "a graph or machinery program" if is_circuit else "a circuit"
        warnings.warn(
            f"{', '.join(ignored)} {'was' if len(ignored) == 1 else 'were'} set, but only {reads} reads "
            f"{'it' if len(ignored) == 1 else 'them'}; this run is {ran}, so "
            f"{'it' if len(ignored) == 1 else 'they'} had no effect.",
            IgnoredOptionWarning,
            # Points at the run() call, which the caller made, rather than at this helper.
            stacklevel=3,
        )

    def _run_local_claveles(self, representation: ReprType) -> SimulatorClientResult:
        from claveles.client._local_simulator_claveles import (  # noqa:PLC0415
            local_run_circuit,
            local_run_graph,
            local_run_machinery,
        )

        # None/"cpu" both map to the simulator's auto-selecting multi-threaded backend; every other
        # value is already a simulator backend name and passes through unchanged.
        claveles_backend = "multi_cpu" if self.backend in {None, "cpu"} else self.backend
        result: ResultType
        shot_results: list[ShotResult] | None = None

        execution_started_at = datetime.now(tz=self.timezone)
        if isinstance(representation, CircuitRepr):
            local_result = local_run_circuit(
                claveles_backend,
                self.n_shots,
                self.state_save_policy,
                representation,
                seed=self.random_seed,
                max_bond_dim=self.max_bond_dim,
                svd_cutoff=self.svd_cutoff,
                track_truncation_error=self.track_truncation_error,
                max_saved_state_elements=self.max_saved_state_elements,
                decompose_std=self.decompose_std,
                state_representation=self.state_representation,
            )
            result = local_result.circuit_result
            shot_results = local_result.shot_results
        elif isinstance(representation, GraphRepr):
            result = local_run_graph(
                claveles_backend, self.n_shots, self._squeezing_level(), representation, seed=self.random_seed
            ).graph_result
        elif isinstance(representation, MachineryRepr):
            result = local_run_machinery(
                claveles_backend, self.n_shots, self._squeezing_level(), representation, seed=self.random_seed
            ).machinery_result
        else:
            # `run` rejects such a value before reaching here, so this is a backstop for a direct call,
            # and it says so with the same wording every other dispatch uses.
            raise TypeError(unsupported_representation_message(representation))
        execution_finished_at = datetime.now(tz=self.timezone)

        return SimulatorClientResult(
            execution_details=ExecutionDetails(
                simulator_version="",
                scheduler_version="",
                timeline=JobTimeline(
                    submitted_at=execution_started_at,
                    execution_started_at=execution_started_at,
                    execution_finished_at=execution_finished_at,
                    finished_at=execution_finished_at,
                ),
            ),
            _result=result,
            _shot_results=shot_results,
        )

    def _run_simulator_server(self, representation: ReprType) -> SimulatorClientResult:
        from claveles.client._simulator_server import run_representation_on_simulator_server  # noqa: PLC0415

        assert self.url is not None  # noqa: S101  # guaranteed by the mode="simulator-server" constructor validation
        assert self.api_key is not None  # noqa: S101  # guaranteed by the mode="simulator-server" constructor validation

        # "cpu" is this client's alias for "multi_cpu" (see BackendType); CSS's own Backend enum
        # has no "cpu" entry, so it must be resolved here, same as `_run_local_claveles` does. Unlike
        # that local-mode mapping, `None` passes through unchanged -- CSS auto-resolves an
        # unspecified backend itself, which is not always "multi_cpu" (e.g. GPU auto-selection).
        server_backend = "multi_cpu" if self.backend == "cpu" else self.backend

        submitted_at = datetime.now(tz=self.timezone)
        server_result = run_representation_on_simulator_server(
            self.url,
            self.api_key,
            representation,
            self.n_shots,
            backend=server_backend,
            seed=self.random_seed,
            timeout_msec=self.timeout_msec,
            resource_squeezing_level=self._squeezing_level(),
            max_bond_dim=self.max_bond_dim,
            svd_cutoff=self.svd_cutoff,
            track_truncation_error=self.track_truncation_error,
            decompose_std=self.decompose_std,
            state_representation=self.state_representation,
            polling_interval=self.polling_interval,
        )
        finished_at = datetime.now(tz=self.timezone)

        # CSS's timestamps parse as UTC-aware (`_parse_timestamp`); convert to `self.timezone` so
        # every `JobTimeline` field shares one timezone, matching `_run_local_claveles`'s timeline.
        def _in_client_tz(value: datetime | None) -> datetime | None:
            return value.astimezone(self.timezone) if value is not None else None

        return SimulatorClientResult(
            execution_details=ExecutionDetails(
                simulator_version=server_result.simulator_version,
                scheduler_version="",
                timeline=JobTimeline(
                    submitted_at=submitted_at,
                    queued_at=_in_client_tz(server_result.queued_at),
                    dequeued_at=_in_client_tz(server_result.started_at),
                    execution_started_at=_in_client_tz(server_result.started_at),
                    execution_finished_at=_in_client_tz(server_result.finished_at),
                    finished_at=finished_at,
                ),
            ),
            _result=server_result.result,
            _shot_results=server_result.shot_results,
        )

    def _run_remote(self, representation: ReprType) -> SimulatorClientResult:
        from claveles.client._simulator_remote import run_representation_remotely  # noqa: PLC0415

        assert self.url is not None  # noqa: S101  # guaranteed by the mode="remote" constructor validation
        assert self.api_key is not None  # noqa: S101  # guaranteed by the mode="remote" constructor validation

        if isinstance(representation, CircuitRepr) and self.state_representation != "auto":
            msg = (
                f"state_representation is {self.state_representation!r}, but mode='remote' has no way "
                "to carry an explicit choice through to the Cloud Simulator Server and only honours "
                "'auto'. Run with mode='simulator-server' or mode='local' to choose one."
            )
            raise ValueError(msg)

        # "cpu" is this client's alias for "multi_cpu" (see BackendType); CSS's own Backend enum
        # has no "cpu" entry, so it must be resolved here, same as `_run_local_claveles`/
        # `_run_simulator_server` do. `None` passes through unchanged -- CSS auto-resolves an
        # unspecified backend itself, which is not always "multi_cpu" (e.g. GPU auto-selection).
        server_backend = "multi_cpu" if self.backend == "cpu" else self.backend

        submitted_at = datetime.now(tz=self.timezone)
        remote_result = run_representation_remotely(
            self.url,
            self.api_key,
            representation,
            self.n_shots,
            backend=server_backend,
            seed=self.random_seed,
            timeout_msec=self.timeout_msec,
            resource_squeezing_level=self._squeezing_level(),
            max_bond_dim=self.max_bond_dim,
            svd_cutoff=self.svd_cutoff,
            track_truncation_error=self.track_truncation_error,
            decompose_std=self.decompose_std,
            state_representation=self.state_representation,
            polling_interval=self.polling_interval,
        )
        finished_at = datetime.now(tz=self.timezone)

        # Both timestamps this path reports are already UTC-aware (one parsed off the wire, one
        # derived from a client-side `datetime.now(tz=...)` -- see `_simulator_remote.py`); convert to
        # `self.timezone` so every `JobTimeline` field shares one timezone, matching
        # `_run_local_claveles`'s and `_run_simulator_server`'s timelines.
        def _in_client_tz(value: datetime | None) -> datetime | None:
            return value.astimezone(self.timezone) if value is not None else None

        return SimulatorClientResult(
            execution_details=ExecutionDetails(
                simulator_version=remote_result.simulator_version,
                scheduler_version="",
                timeline=JobTimeline(
                    submitted_at=submitted_at,
                    queued_at=_in_client_tz(remote_result.queued_at),
                    dequeued_at=_in_client_tz(remote_result.started_at),
                    execution_started_at=_in_client_tz(remote_result.started_at),
                    execution_finished_at=_in_client_tz(remote_result.finished_at),
                    finished_at=finished_at,
                ),
            ),
            _result=remote_result.result,
            _shot_results=remote_result.shot_results,
        )

    def run(self, representation: ReprType) -> SimulatorClientResult:
        """Run a representation with the simulator.

        Note:
            ``mode="local"``, ``mode="simulator-server"`` and ``mode="remote"`` all support circuit,
            graph, and machinery representations.

        Args:
            representation (ReprType): Representation.

        Returns:
            SimulatorClientResult: Result of the running.

        Raises:
            TypeError: If ``representation`` is not a ``CircuitRepr``, a ``GraphRepr`` or a
                ``MachineryRepr``. Every mode refuses it the same way, before anything is run or
                uploaded.
            ValueError: If an option was assigned a value this mode cannot honour.

        Warns:
            IgnoredOptionWarning: If an option was set that this representation does not read.

        Example:
            >>> from numpy import pi
            >>> from math import pi, sqrt
            >>> from claveles.client import SimulatorClient
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> from claveles.circuit.state import BosonicState
            >>> client = SimulatorClient(n_shots=1, backend="cpu")
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | BosonicState.vacuum()
            [QuMode(id=0)]
            >>> circuit.Q(0) | std.PhaseRotation(pi / 2)
            [QuMode(id=0)]
            >>> circuit.Q(0) | std.MeasureHomodyne(pi / 2)
            Var([std.measure_homodyne] [1.5707963267948966] [0])
            >>> result = client.run(circuit)
        """  # noqa: DOC502  # ValueError propagates from _check_mode_can_still_honour_its_options
        # Checked here, before dispatching, for two reasons: every mode then refuses the same value the
        # same way rather than each deciding for itself, and a rejected call does not first import the
        # simulator wheel, which `mode="local"` needs for a run but not to say no.
        check_representation(representation)
        self._check_mode_can_still_honour_its_options()
        self._warn_about_options_this_run_will_not_read(representation)
        if self.mode == "simulator-server":
            return self._run_simulator_server(representation)
        if self.mode == "remote":
            return self._run_remote(representation)
        return self._run_local_claveles(representation)
