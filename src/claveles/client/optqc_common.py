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

"""Common classes and utilities for OptQC clients.

This module contains base classes, compiler/decoder utilities, and data
containers shared across all OptQC client implementations (Cloud and Scheduler).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any, Literal

import requests

from claveles.circuit.result import CircuitResult
from claveles.client.abstract import (
    AbstractClient,
    AbstractClientResult,
    ReprType,
    ResultType,
)
from claveles.client.exceptions import (
    ClientAuthError,
    ClientRequestError,
    ClientServerError,
    JobFailedError,
    JobTimeoutError,
)
from claveles.graph.convert import (
    BeamSearchConverter,
    BeamSearchConvertSettings,
    GreedyConverter,
    GreedyConvertSettings,
)
from claveles.graph.result import GraphResult
from claveles.machinery import MachineryRepr

if TYPE_CHECKING:
    from datetime import timedelta

    from claveles.circuit import CircuitRepr
    from claveles.graph import GraphRepr
    from claveles.machinery.result import MachineryResult

OPTQC_N_LOCAL_MACRONODES = 129
"""Default number of local macronodes for the OptQC hardware."""

logger = logging.getLogger(__name__)

_POLLING_TIMEOUT_MARGIN_MSEC = 600_000
"""Extra margin (ms) added on top of timeout_msec to derive the default polling deadline."""


class JobStatus(str, Enum):  # noqa: UP042  # enum.StrEnum is 3.11+, and this package supports 3.10
    """The state a submitted job is in, as one fixed vocabulary.

    The wire carries a status as a string, and more than one spelling reaches the SDK for the same
    state: ``COMPLETED`` for a job that succeeded, ``FAILED`` for one that failed, ``CANCELED`` for
    one that was cancelled, in either case. Those spellings are recognized on the way in
    (:meth:`from_wire`) and collapse onto the members below, so a caller compares against one name
    rather than guessing which synonym a route happens to use, and the SDK has one place where the
    vocabulary is written down.

    A ``str`` subclass so that ``status == "SUCCESS"`` still reads true; use :attr:`value` when
    interpolating into a message, since formatting an enum member yields ``JobStatus.SUCCESS``.
    """

    SUBMITTED = "SUBMITTED"
    """Accepted by the cloud API, waiting for the bridge to pick it up.

    A state the cloud route passes through before :attr:`QUEUED`: the job is registered but no
    worker has acknowledged it yet, so a bridge that has stopped leaves a job here. Unlike
    ``QUEUED``, it names no queue position, and the submission route never reports it.
    """

    QUEUED = "QUEUED"
    """Accepted and waiting to run."""

    RUNNING = "RUNNING"
    """Executing now."""

    SUCCESS = "SUCCESS"
    """Finished, with a result to download."""

    FAILURE = "FAILURE"
    """Finished without a result, because execution failed."""

    TIMEOUT = "TIMEOUT"
    """Stopped by the server because it exceeded its execution timeout."""

    CANCELLED = "CANCELLED"
    """Stopped before finishing, at someone's request."""

    @classmethod
    def from_wire(cls, status: str) -> JobStatus:
        """Return the member a status string off the wire names.

        Args:
            status: The status exactly as the server reported it. Case and surrounding whitespace are
                ignored, and the recognized synonyms are folded onto their member.

        Returns:
            JobStatus: The state the string names.

        Raises:
            ClientServerError: If the string names no known state. Reported rather than treated as
                "not finished yet", because a status this SDK cannot interpret is a version skew
                between it and the server, and guessing would mean waiting out the polling deadline
                on a job that already finished.
        """
        normalized = status.strip().upper()
        synonym = _WIRE_STATUS_SYNONYMS.get(normalized)
        if synonym is not None:
            return synonym
        try:
            return cls(normalized)
        except ValueError:
            known = ", ".join(sorted({*(member.value for member in cls), *_WIRE_STATUS_SYNONYMS}))
            msg = (
                f"The server reported job status {status!r}, which this version of the SDK does not "
                f"know. Known statuses are: {known}. Upgrading claveles is the likely fix."
            )
            raise ClientServerError(msg) from None

    @property
    def is_terminal(self) -> bool:
        """Whether the job has stopped, so polling it again cannot change the answer.

        Returns:
            bool: True for a state the job does not leave.
        """
        return self in _TERMINAL_STATUSES

    @property
    def is_success(self) -> bool:
        """Whether the job finished with a result to download.

        Returns:
            bool: True only for :attr:`SUCCESS`.
        """
        return self is JobStatus.SUCCESS


_WIRE_STATUS_SYNONYMS: dict[str, JobStatus] = {
    "COMPLETED": JobStatus.SUCCESS,
    "FAILED": JobStatus.FAILURE,
    "CANCELED": JobStatus.CANCELLED,
}
"""Spellings that reach the SDK for a state :class:`JobStatus` already has a name for."""

_TERMINAL_STATUSES: frozenset[JobStatus] = frozenset({
    JobStatus.SUCCESS,
    JobStatus.FAILURE,
    JobStatus.TIMEOUT,
    JobStatus.CANCELLED,
})
"""States a job does not leave. ``SUBMITTED`` is deliberately absent: it is a waiting state."""


class OptQCGraphConversionAlgo(Enum):
    """Algorithm for converting circuit representations to graph representations."""

    Greedy = 0
    BeamSearch = 1


@dataclass
class OptQCCompiler:
    """Compiler for converting between quantum circuit representations.

    This class encapsulates the full compile pipeline used by
    :class:`~claveles.client.optqc_client.OptQCClient`:

    1. ``CircuitRepr`` -> ``GraphRepr`` via a selected graph conversion algorithm.
    2. ``GraphRepr`` -> ``MachineryRepr`` with target-hardware local-macronode constraints.

    The compiler is configurable and can be passed to ``OptQCClient(compiler=...)``
    to tune conversion behavior for latency/quality trade-offs and hardware handling.
    """

    n_local_macronodes: int = OPTQC_N_LOCAL_MACRONODES
    """Target number of local macronodes expected by hardware.

    Graphs smaller than this value are padded during :meth:`graph_to_machinery`; larger graphs raise
    ``ValueError``.
    """

    graph_conversion_algo: OptQCGraphConversionAlgo = OptQCGraphConversionAlgo.BeamSearch
    """Algorithm for circuit-to-graph conversion.

    ``BeamSearch`` generally yields better solutions at higher cost, while ``Greedy`` is faster and
    simpler.
    """

    beam_width: int = 10
    """Beam width used when :attr:`graph_conversion_algo` is ``BeamSearch``.

    Larger values explore more candidates.
    """

    empty_wire_policy: Literal["teleport", "measure"] = "teleport"
    """Policy for handling padded/empty macronodes when converting to machinery representation.

    Accepted values are ``"teleport"`` and ``"measure"``.
    """

    def circuit_to_graph(self, circuit: CircuitRepr) -> GraphRepr:
        """Convert a circuit representation to a graph representation.

        The conversion behavior depends on ``self.graph_conversion_algo``:
        - ``OptQCGraphConversionAlgo.Greedy``: lower overhead, local decisions.
        - ``OptQCGraphConversionAlgo.BeamSearch``: global search with ``beam_width``.

        Args:
            circuit: Source circuit representation.

        Returns:
            Converted graph representation.

        Raises:
            ValueError: If the circuit asks for something a graph representation cannot express, such as
                post-selection, or if the conversion algorithm is not supported. What a graph can carry
                is decided by the conversion itself, so that this and a conversion a caller runs
                directly refuse the same circuits.
        """
        if self.graph_conversion_algo == OptQCGraphConversionAlgo.Greedy:
            settings = GreedyConvertSettings(self.n_local_macronodes)
            converter = GreedyConverter(settings)
            return converter.convert(circuit)
        if self.graph_conversion_algo == OptQCGraphConversionAlgo.BeamSearch:
            settings = BeamSearchConvertSettings(self.n_local_macronodes, beam_width=self.beam_width)
            converter = BeamSearchConverter(settings)
            return converter.convert(circuit)

        msg = f"Unsupported graph conversion algorithm: {self.graph_conversion_algo}"
        raise ValueError(msg)

    def graph_to_machinery(self, graph: GraphRepr) -> MachineryRepr:
        """Convert a graph representation to a machinery representation.

        If the graph has fewer local macronodes than the target hardware,
        it is padded to match. If it has more, a ``ValueError`` is raised.
        The resulting machinery conversion uses ``self.empty_wire_policy``.

        Args:
            graph: Source graph representation.

        Returns:
            Converted machinery representation.

        Raises:
            ValueError: If the graph has more local macronodes than the target hardware.
        """
        if self.n_local_macronodes < graph.n_local_macronodes:
            msg = (
                f"Graph has {graph.n_local_macronodes} local macronodes, "
                f"but the target hardware only supports {self.n_local_macronodes}."
            )
            raise ValueError(msg)

        if graph.n_local_macronodes < self.n_local_macronodes:
            graph = deepcopy(graph)
            graph.increase_local_macronodes(self.n_local_macronodes)
        return MachineryRepr.from_graph_repr(graph, empty_wire_policy=self.empty_wire_policy)


class OptQCResultDecoder:
    """Decoder for converting raw hardware results back to higher-level representations."""

    @staticmethod
    def machinery_to_graph(machinery_result: MachineryResult, machinery_input: MachineryRepr) -> GraphResult:
        """Convert a machinery result to a graph result.

        Args:
            machinery_result: Raw machinery execution result.
            machinery_input: The machinery representation that was executed.

        Returns:
            Decoded graph result.
        """
        return GraphResult.construct_from_machinery_result(machinery_result, machinery_input)

    @staticmethod
    def graph_to_circuit(graph_result: GraphResult, graph_input: GraphRepr) -> CircuitResult:
        """Convert a graph result to a circuit result.

        Args:
            graph_result: Decoded graph result.
            graph_input: The graph representation that was executed.

        Returns:
            Decoded circuit result.
        """
        return CircuitResult.construct_from_graph_result(graph_result, graph_input)


class OptQCClientResult(AbstractClientResult):
    """Result of an OptQC job execution.

    Holds the execution result at multiple representation levels
    (circuit, graph, machinery). Which levels are populated depends on
    the input representation type used when submitting the job.
    """

    circuit_result: CircuitResult | None
    """Result in circuit representation, or None if the job was not submitted as a circuit.

    Only a circuit submission leaves behind the circuit and the graph this level is decoded from, so
    this is the narrowest of the three.
    """

    graph_result: GraphResult | None
    """Result in graph representation, or None if the job was submitted as a machinery representation.

    Decoding this level needs the graph the job was compiled from, which a circuit and a graph
    submission both leave behind and a machinery submission does not -- so a circuit or a graph fills
    this in, and a machinery representation is the only one that does not.
    """

    machinery_result: MachineryResult | None
    """Result in machinery representation.

    What the hardware reports, and what the other two levels are decoded from, so
    :meth:`~claveles.client.optqc_client.OptQCClient.get_job_result` always fills this one in.
    """

    details: dict[str, Any]
    """Additional execution details, such as timing information.

    Empty rather than None when there are none, which is currently always: the job endpoints report a
    status and a result location and no timing, the same reason :attr:`execution_time` is None. Use
    :attr:`~claveles.execute.ExecutionResult.total_time` for the duration of the call itself.
    """

    def __init__(
        self,
        circuit_result: CircuitResult | None = None,
        graph_result: GraphResult | None = None,
        machinery_result: MachineryResult | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Constructor.

        Args:
            circuit_result: Result in circuit representation.
            graph_result: Result in graph representation.
            machinery_result: Result in machinery representation.
            details: Additional execution details (e.g. timing information).
        """
        self.circuit_result = circuit_result
        self.graph_result = graph_result
        self.machinery_result = machinery_result
        self.details = details or {}

    @property
    def execution_time(self) -> timedelta | None:
        """Return the execution time the API reported, which is currently never.

        The job endpoints this client reads (``/jobs/{id}`` on either route) report a status and a
        result location, and no timing. Rather than substitute the round trip measured on this side,
        which would be a different quantity, this reports None until the API carries one. Use
        :attr:`~claveles.execute.ExecutionResult.total_time` for the duration of the call itself.

        Returns:
            timedelta | None: Always None.
        """
        return None

    @property
    def execution_result(self) -> ResultType:
        """Return the highest-level available execution result.

        Priority order: circuit > graph > machinery.

        Raises:
            ValueError: If no execution result is available.
        """
        if self.circuit_result is not None:
            return self.circuit_result
        if self.graph_result is not None:
            return self.graph_result
        if self.machinery_result is not None:
            return self.machinery_result

        msg = "No execution result available."
        raise ValueError(msg)


@dataclass
class OptQCSubmissionContext:
    """Context object returned after job submission.

    Stores the job ID along with the intermediate representations used
    during compilation. These are needed later when decoding the raw
    hardware result back to the original representation level.
    """

    job_id: str
    circuit_repr: CircuitRepr | None = field(default=None, repr=False)
    graph_repr: GraphRepr | None = field(default=None, repr=False)
    machinery_repr: MachineryRepr | None = field(default=None, repr=False)


class OptQCClientBase(AbstractClient):
    """Base class for OptQC clients.

    Provides common HTTP request handling, error classification, and
    the synchronous ``run()`` polling loop.
    """

    polling_interval: float
    """Seconds between job status requests while waiting for a job to finish.

    Read on each pass of :meth:`wait_for_completion`, so assigning to it changes the interval of a
    loop that is already running from the next pass onwards.
    """

    def __init__(  # noqa: PLR0913
        self,
        url: str,
        api_key: str,
        n_shots: int = 1000,
        timeout_msec: int = 10_000,
        *,
        polling_interval: float,
        polling_timeout_msec: int | None = None,
        compiler: OptQCCompiler,
    ) -> None:
        """Constructor.

        Args:
            url: API base URL (trailing slashes are stripped).
            api_key: API key for API authentication.
            n_shots: Number of measurement shots per job. Defaults to 1000.
            timeout_msec: Job execution timeout in milliseconds sent to the server. Defaults to 10,000 (10 seconds).
            polling_interval: Interval in seconds between job status polling requests.
            polling_timeout_msec: Client-side polling deadline in milliseconds. When ``None`` (default),
                defaults to ``timeout_msec + 600_000`` (10 minutes of extra margin). Pass ``0`` to
                disable the deadline entirely (poll indefinitely).
            compiler: Compiler for converting between representations.
        """
        super().__init__(n_shots)
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout_msec = timeout_msec
        self.polling_interval = polling_interval
        self._polling_timeout_msec: int = (
            timeout_msec + _POLLING_TIMEOUT_MARGIN_MSEC if polling_timeout_msec is None else polling_timeout_msec
        )
        self._compiler = compiler

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:  # noqa: ANN401
        """Send an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PUT, etc.).
            path: API endpoint path (leading slashes are handled).
            **kwargs: Additional keyword arguments passed to ``requests.request``.

        Returns:
            Response object for successful requests (2xx/3xx).

        Raises:
            ClientServerError: If the server returns a 5xx status code or the connection fails.
            ClientAuthError: If authentication fails (401/403).
            ClientRequestError: If the server returns a 4xx client error.
        """
        url = f"{self._url}/{path.lstrip('/')}"
        if "timeout" not in kwargs:
            kwargs["timeout"] = 30

        try:
            response = requests.request(method, url, headers=self._headers, **kwargs)  # noqa: S113
        except requests.RequestException as e:
            msg = f"Failed to connect to {url}: {e}"
            raise ClientServerError(msg) from e

        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            msg = f"Server error {response.status_code}: {response.text}"
            raise ClientServerError(msg)
        if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            msg = f"Authentication error {response.status_code}: {response.text}"
            raise ClientAuthError(msg)
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            msg = f"Client error {response.status_code}: {response.text}"
            raise ClientRequestError(msg)

        return response

    @abstractmethod
    def health(self) -> bool:
        """Check the health of the API service.

        Returns:
            ``True`` if the service is healthy.
        """
        raise NotImplementedError

    @abstractmethod
    def version(self) -> dict[str, str]:
        """Get the service versions.

        Returns:
            dict[str, str]: A dictionary containing version information.
                For cloud targets: ``{"cloud_version": "...", "user_api_version": "..."}``.
                For direct targets:
                ``{"scheduler_version": "...", "submission_api_version": "..."}``.
        """
        raise NotImplementedError

    @abstractmethod
    def list_qc_nodes(self) -> list[dict[str, Any]]:
        """List available QC nodes.

        Returns:
            List of QC node information dictionaries.
        """
        raise NotImplementedError

    @abstractmethod
    def submit_job(self, representation: ReprType) -> OptQCSubmissionContext:
        """Submit a job for asynchronous execution.

        Args:
            representation: Quantum program in any supported representation.

        Returns:
            Submission context for tracking the job.
        """
        raise NotImplementedError

    @abstractmethod
    def get_job_status(self, context: OptQCSubmissionContext) -> JobStatus:
        """Get the current status of a submitted job.

        Args:
            context: Submission context returned by ``submit_job``.

        Returns:
            JobStatus: The state the job is in. Normalized on the way in, so this and
                :meth:`wait_for_completion` report a state under one spelling.
        """
        raise NotImplementedError

    @abstractmethod
    def get_job_result(self, context: OptQCSubmissionContext) -> OptQCClientResult:
        """Get the result of a completed job.

        Args:
            context: Submission context returned by ``submit_job``.

        Returns:
            Decoded job result at all available representation levels.

        Raises:
            ClientServerError: If the job has not completed successfully or the result cannot be
                retrieved or decoded.
        """
        raise NotImplementedError

    def wait_for_completion(self, context: OptQCSubmissionContext) -> JobStatus:
        """Poll job status until it reaches a terminal state.

        Args:
            context: Submission context returned by ``submit_job``.

        Returns:
            JobStatus: The terminal state the job reached.

        Raises:
            JobTimeoutError: If the polling deadline (``polling_timeout_msec``) is exceeded before the
                job reaches a terminal state. Also a ``TimeoutError``, and the same class the
                simulator-server path raises for its own deadline.
        """
        deadline: float | None = None
        if self._polling_timeout_msec > 0:
            deadline = monotonic() + self._polling_timeout_msec / 1000
            logger.info(
                "Waiting for job %s (polling timeout: %.1f min, interval: %.1f s)",
                context.job_id,
                self._polling_timeout_msec / 60_000,
                self.polling_interval,
            )
        else:
            logger.info(
                "Waiting for job %s (no polling timeout, interval: %.1f s)",
                context.job_id,
                self.polling_interval,
            )

        prev_status: JobStatus | None = None
        while True:
            if deadline is not None and monotonic() >= deadline:
                msg = (
                    f"Job {context.job_id} did not complete within the polling timeout "
                    f"({self._polling_timeout_msec} ms). The job may still be running on the server."
                )
                raise JobTimeoutError(msg)

            status = self.get_job_status(context)

            if status != prev_status:
                logger.info("Job %s status: %s", context.job_id, status.value)
                prev_status = status

            if status.is_terminal:
                return status

            sleep(self.polling_interval)

    def run(self, representation: ReprType) -> OptQCClientResult:
        """Submit a job and wait for the result synchronously.

        Submits the job, polls for completion, and returns the decoded result.
        The polling interval is controlled by ``self.polling_interval``.

        Args:
            representation: Quantum program in any supported representation.

        Returns:
            Decoded job result.

        Raises:
            JobFailedError: If the job reaches a terminal state other than
                :attr:`JobStatus.SUCCESS`.
            JobTimeoutError: If the wait runs out before the job reaches a terminal state.
        """  # noqa: DOC502  # JobTimeoutError propagates from wait_for_completion
        context = self.submit_job(representation)
        status = self.wait_for_completion(context)
        if not status.is_success:
            msg = f"Job {context.job_id} failed with status {status.value}."
            raise JobFailedError(msg)

        return self.get_job_result(context)
