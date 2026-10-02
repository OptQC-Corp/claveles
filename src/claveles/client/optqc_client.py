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

"""OptQC unified client.

This module provides :class:`OptQCClient`, a unified client that supports
both the Cloud API and the Submission API (direct
connection to the scheduler). The routing is determined by the ``target``
parameter at construction time.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Literal

import requests

from claveles.circuit import CircuitRepr, CircuitResult
from claveles.client.abstract import check_representation
from claveles.client.exceptions import ClientServerError
from claveles.client.optqc_common import (
    JobStatus,
    OptQCClientBase,
    OptQCClientResult,
    OptQCCompiler,
    OptQCResultDecoder,
    OptQCSubmissionContext,
)
from claveles.client.optqc_rest_adapter import (
    json_to_machinery_result,
    machinery_repr_to_json,
)
from claveles.graph import GraphRepr, GraphResult
from claveles.machinery import MachineryRepr

if TYPE_CHECKING:
    from claveles.client.abstract import ReprType
    from claveles.machinery.result import MachineryResult

OptQCTarget = Literal["cloud", "direct"]
"""Which route :class:`OptQCClient` submits over.

- ``cloud``: the Cloud User API (``/api/user/...``).
- ``direct``: a direct scheduler connection over the Submission API (``/v1/direct/...``).
"""

# Maps the logical routing target accepted by ``OptQCClient`` to the path
# segment of the scheduler Submission API URL (``/v1/<segment>/...``). This
# indirection decouples the public target name from the wire path so one can
# change without the other, and its keys are the set of valid targets: a
# target absent here is rejected rather than forming an unknown URL.
_SCHEDULER_TARGET_PATH_SEGMENTS: dict[str, str] = {
    "direct": "direct",
}

_CLOUD_TARGET = "cloud"
"""The one target that does not go through the scheduler Submission API."""

_VALID_TARGETS: frozenset[str] = frozenset({_CLOUD_TARGET, *_SCHEDULER_TARGET_PATH_SEGMENTS})
"""Every target the constructor accepts, derived from where each one routes."""


def _decode_machinery_result(content: bytes) -> MachineryResult:
    """Parse a result payload, reporting anything unparseable as a server error.

    Both routes download the same JSON body and hand it to
    :func:`~claveles.client.optqc_rest_adapter.json_to_machinery_result`, which indexes into it and so
    raises whatever a mismatched payload happens to produce -- a ``KeyError`` for a missing field, a
    ``ValueError`` for a shot whose values do not group into fours, an ``IndexError`` for a readout
    index the header does not cover. None of those are anything the caller did, and the documented
    contract of ``get_job_result`` is a single error type, so they are reported as one here rather than
    escaping raw from whichever route downloaded the body.

    Args:
        content: The raw response body, expected to be JSON.

    Returns:
        MachineryResult: The decoded result.

    Raises:
        ClientServerError: If the body is not JSON, or is JSON that is not a machinery result.
    """
    # ValueError covers json.JSONDecodeError, which subclasses it, so a body that is not JSON at all
    # and a body that is the wrong JSON are reported the same way.
    try:
        return json_to_machinery_result(json.loads(content))
    except (KeyError, IndexError, TypeError, ValueError) as e:
        msg = f"Failed to decode result data: {type(e).__name__}: {e}"
        raise ClientServerError(msg) from e


@dataclass(frozen=True)
class _JobSettings:
    """What to run a job with, as read off the client at the moment of submission."""

    n_shots: int
    timeout_msec: int

    def as_payload(self) -> dict[str, int]:
        """Return the ``settings`` object both routes send.

        Returns:
            dict[str, int]: The settings, spelled as the wire spells them.
        """
        return {"n_shots": self.n_shots, "timeout_msec": self.timeout_msec}


class _OptQCBackend(ABC):
    """Abstract base class for OptQC API backends.

    Each backend encapsulates the HTTP interaction details for a specific
    API route (Cloud or Scheduler). Backend instances delegate HTTP calls
    to the owning :class:`OptQCClient` via ``self._request``.
    """

    def __init__(self, request_fn: Any) -> None:  # noqa: ANN401
        """Constructor.

        Args:
            request_fn: Bound ``_request`` method of the owning client.
        """
        self._request = request_fn

    @abstractmethod
    def health(self) -> bool:
        """Check the health of the service."""

    @abstractmethod
    def version(self) -> dict[str, str]:
        """Get the service versions."""

    @abstractmethod
    def list_qc_nodes(self) -> list[dict[str, Any]]:
        """List available QC nodes."""

    @abstractmethod
    def submit_job(self, machinery_repr: MachineryRepr, settings: _JobSettings) -> str:
        """Submit a job and return the job ID.

        Args:
            machinery_repr: Compiled machinery representation.
            settings: What to run it with, read off the client at this moment rather than copied when
                the backend was built -- ``client.n_shots`` is public and assignable, and a backend
                holding a stale copy meant a later assignment changed nothing on the wire.

        Returns:
            Job ID assigned by the server.
        """

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            job_id: Job identifier.

        Returns:
            JobStatus: The state the job is in, normalized onto the one vocabulary.
        """

    @abstractmethod
    def get_job_result(self, job_id: str) -> MachineryResult:
        """Get the result of a completed job.

        Args:
            job_id: Job identifier.

        Returns:
            Decoded machinery result.
        """


class _OptQCSchedulerBackend(_OptQCBackend):
    """Backend for direct connections to the scheduler.

    Uses the Submission API (``/v1/direct/...``) endpoints.
    """

    def __init__(
        self,
        request_fn: Any,  # noqa: ANN401
        target: str,
        qc_node_id: str,
    ) -> None:
        """Constructor.

        Args:
            request_fn: Bound ``_request`` method of the owning client.
            target: Logical routing target (``"direct"``), mapped internally to
                the scheduler's actual API path segment.
            qc_node_id: Target QC node identifier.

        Raises:
            ValueError: If ``target`` is not a recognized routing target.
        """
        super().__init__(request_fn)
        path_segment = _SCHEDULER_TARGET_PATH_SEGMENTS.get(target)
        if path_segment is None:
            msg = f"Unknown target '{target}'. Expected one of {sorted(_SCHEDULER_TARGET_PATH_SEGMENTS)}."
            raise ValueError(msg)
        self._path_prefix = f"v1/{path_segment}"
        self._qc_node_id = qc_node_id

    def health(self) -> bool:
        """Check the health of the scheduler service.

        Returns:
            ``True`` if the service is healthy.
        """
        response = self._request("GET", f"{self._path_prefix}/healthz")
        return response.status_code == HTTPStatus.OK

    def version(self) -> dict[str, str]:
        """Get the scheduler service version.

        Returns:
            dict[str, str]: Version information.
                e.g. ``{"scheduler_version": "...", "submission_api_version": "..."}``
        """
        response = self._request("GET", f"{self._path_prefix}/version")
        data = response.json()
        if "version" in data:
            data["scheduler_version"] = data.pop("version")
        if "api" in data:
            data["submission_api_version"] = data.pop("api")
        return data

    def list_qc_nodes(self) -> list[dict[str, Any]]:
        """List available QC nodes from the scheduler.

        Returns:
            List of QC node information dictionaries.
        """
        response = self._request("GET", f"{self._path_prefix}/qc_nodes")
        return response.json()

    def submit_job(self, machinery_repr: MachineryRepr, settings: _JobSettings) -> str:
        """Submit a job to the scheduler.

        Args:
            machinery_repr: Compiled machinery representation.
            settings: What to run it with.

        Returns:
            Job ID assigned by the scheduler.

        Raises:
            ClientServerError: If the server returns an unexpected status code.
        """
        payload = {
            "qc_node_id": self._qc_node_id,
            "program": machinery_repr_to_json(machinery_repr),
            "settings": settings.as_payload(),
        }

        response = self._request("POST", f"{self._path_prefix}/jobs", json=payload)

        if response.status_code == HTTPStatus.ACCEPTED:
            data = response.json()
            return data["job_id"]

        msg = f"Unexpected status code {response.status_code}: {response.text}"
        raise ClientServerError(msg)

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a scheduler job.

        Args:
            job_id: Job identifier.

        Returns:
            JobStatus: The state the job is in.
        """
        response = self._request("GET", f"{self._path_prefix}/jobs/{job_id}")
        data = response.json()
        return JobStatus.from_wire(data["status"])

    def get_job_result(self, job_id: str) -> MachineryResult:
        """Download and decode the result of a completed scheduler job.

        The result is returned as JSON from the ``/jobs/{job_id}/result``
        endpoint (transparently gzip-decoded by the HTTP client based on the
        server's ``Content-Encoding`` header).

        Args:
            job_id: Job identifier.

        Returns:
            Decoded machinery result.

        Raises:
            ClientServerError: If the job did not succeed, or the result cannot be downloaded or
                decoded.
        """
        # Checked before downloading, the same way the cloud route checks it: asking for the result of
        # a job that failed is one mistake, so it is reported the same way on both routes rather than as
        # a decoding failure on whichever target the client happens to be built with.
        status = self.get_job_status(job_id)
        if not status.is_success:
            msg = f"Job is not successful (status: {status.value})."
            raise ClientServerError(msg)

        response = self._request("GET", f"{self._path_prefix}/jobs/{job_id}/result")
        return _decode_machinery_result(response.content)


class _OptQCCloudBackend(_OptQCBackend):
    """Backend for the OptQC Cloud API.

    Uses the User API (``/api/user/...``) endpoints, which involve
    S3 presigned URLs for program upload and result download.
    """

    def __init__(self, request_fn: Any) -> None:  # noqa: ANN401
        """Constructor.

        Args:
            request_fn: Bound ``_request`` method of the owning client.
        """
        super().__init__(request_fn)

    def health(self) -> bool:
        """Check the health of the Cloud API.

        Returns:
            ``True`` if the service is healthy.
        """
        response = self._request("GET", "api/user/healthz")
        return response.status_code == HTTPStatus.OK

    def version(self) -> dict[str, str]:
        """Get the Cloud API version.

        Returns:
            dict[str, str]: Version information.
                e.g. ``{"cloud_version": "...", "user_api_version": "..."}``
        """
        response = self._request("GET", "api/user/version")
        data = response.json()
        if "version" in data:
            data["cloud_version"] = data.pop("version")
        if "api" in data:
            data["user_api_version"] = data.pop("api")
        return data

    def list_qc_nodes(self) -> list[dict[str, Any]]:
        """List available QC nodes from the Cloud API.

        Returns:
            List of QC node information dictionaries.
        """
        response = self._request("GET", "api/user/v1/qc_nodes")
        return response.json()

    def submit_job(self, machinery_repr: MachineryRepr, settings: _JobSettings) -> str:
        """Submit a job via the Cloud API.

        This is a three-step process:
        1. Request an upload URL and a job ID.
        2. Upload the program JSON to S3 via the presigned URL.
        3. Register the job with execution settings.

        Args:
            machinery_repr: Compiled machinery representation.
            settings: What to run it with.

        Returns:
            Job ID assigned by the Cloud API.

        Raises:
            ClientServerError: If the upload or job registration fails.
        """
        # Step 1: Get upload URL and job ID
        response = self._request("POST", "api/user/v1/jobs")
        upload_data = response.json()
        job_id = upload_data["job_id"]
        upload_url = upload_data["upload_url"]

        # Step 2: Upload program to S3
        program_json = machinery_repr_to_json(machinery_repr)
        headers = {"Content-Type": "application/json", "Host": urllib.parse.urlparse(upload_url).netloc}
        try:
            upload_response = requests.put(upload_url, data=json.dumps(program_json), headers=headers, timeout=30)
            upload_response.raise_for_status()
        except requests.RequestException as e:
            msg = f"Failed to upload program to S3: {e}"
            raise ClientServerError(msg) from e

        # Step 3: Register the job
        payload = {"settings": settings.as_payload()}
        self._request("POST", f"api/user/v1/jobs/{job_id}/submit", json=payload)
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a Cloud API job.

        Args:
            job_id: Job identifier.

        Returns:
            JobStatus: The state the job is in.
        """
        response = self._request("GET", f"api/user/v1/jobs/{job_id}")
        data = response.json()
        return JobStatus.from_wire(data["status"])

    def get_job_result(self, job_id: str) -> MachineryResult:
        """Download and decode the result of a completed Cloud API job.

        The result is downloaded from S3 via a presigned URL provided
        in the job details response.

        Args:
            job_id: Job identifier.

        Returns:
            Decoded machinery result.

        Raises:
            ClientServerError: If the job is not successful, the download URL
                is missing, or the result cannot be downloaded/decoded.
        """
        response = self._request("GET", f"api/user/v1/jobs/{job_id}")
        data = response.json()

        status = JobStatus.from_wire(data["status"])
        if not status.is_success:
            msg = f"Job is not successful (status: {status.value})."
            raise ClientServerError(msg)

        download_url = data.get("download_url")
        if not download_url:
            msg = "Download URL not found in job details."
            raise ClientServerError(msg)

        try:
            res = requests.get(download_url, timeout=30)
            res.raise_for_status()
        except requests.RequestException as e:
            msg = f"Failed to download result: {e}"
            raise ClientServerError(msg) from e

        return _decode_machinery_result(res.content)


class OptQCClient(OptQCClientBase):
    """Unified client for OptQC quantum computers.

    Supports both the Cloud API and the Submission API (direct scheduler
    connection). The routing is determined by the ``target`` parameter.

    Example:
        Cloud connection::

            client = OptQCClient(
                url="https://compute.cloud.optqc.com",
                api_key="your-api-key",
                target="cloud",
            )

        Direct scheduler connection::

            client = OptQCClient(
                url="https://scheduler.local",
                api_key="your-api-key",
                target="direct",
                qc_node_id="node-001",
            )
    """

    def __init__(  # noqa: PLR0913
        self,
        url: str | None = None,
        api_key: str | None = None,
        target: OptQCTarget | None = None,
        *,
        qc_node_id: str | None = None,
        n_shots: int = 1000,
        timeout_msec: int = 10_000,
        polling_interval: float = 1.0,
        polling_timeout_msec: int | None = None,
        compiler: OptQCCompiler | None = None,
    ) -> None:
        """Constructor.

        Args:
            url: API base URL. If not specified, it is retrieved from the environment variable ``"OPTQC_CLIENT_URL"``.
            api_key: API key for API authentication.
                If not specified, it is retrieved from the environment variable ``"OPTQC_CLIENT_API_KEY"``.
            target (:data:`OptQCTarget` | None): Routing target. Use ``"cloud"`` for the Cloud API, or
                ``"direct"`` for a direct scheduler connection.
                If not specified, it is retrieved from the environment variable ``"OPTQC_CLIENT_TARGET"``.
                If neither is provided, it defaults to ``"cloud"``.
            qc_node_id: Target QC node identifier. Required for direct
                scheduler connections (``target != "cloud"``).
            n_shots: Number of measurement shots per job. Defaults to 1000.
            timeout_msec: Job execution timeout in milliseconds sent to the server. Defaults to 10,000 (10 seconds).
            polling_interval: Interval in seconds between job status polls. Defaults to 1.0.
            polling_timeout_msec: Client-side polling deadline in milliseconds. When ``None`` (default),
                defaults to ``timeout_msec + 600_000`` (10 minutes of extra margin). Pass ``0`` to
                poll indefinitely.
            compiler: Compiler for representation conversion.
                If ``None``, a default compiler is used.

        Raises:
            ValueError: If ``target`` is not one of ``"cloud"``, ``"direct"``, if the URL or API key is
                missing, or if ``target`` is not ``"cloud"`` and ``qc_node_id`` is not provided.
        """
        resolved_url = url or os.getenv("OPTQC_CLIENT_URL", "")
        resolved_api_key = api_key or os.getenv("OPTQC_CLIENT_API_KEY", "")
        resolved_target = target or os.getenv("OPTQC_CLIENT_TARGET", _CLOUD_TARGET)

        # Everything the client needs to form a request is checked here, before the object exists, and
        # in the order a reader would ask about it: where to send, what to send it with, and what the
        # chosen route additionally needs. Checking the target last would report an unknown one as a
        # missing qc_node_id, and leaving the URL and key unchecked would surface much later as a failure
        # to connect to a path under the empty string.
        if resolved_target not in _VALID_TARGETS:
            msg = f"Unknown target {resolved_target!r}. Expected one of {sorted(_VALID_TARGETS)}."
            raise ValueError(msg)
        if not resolved_url or not resolved_api_key:
            msg = (
                "OptQCClient requires url and api_key, either passed directly or via the "
                "OPTQC_CLIENT_URL / OPTQC_CLIENT_API_KEY environment variables."
            )
            raise ValueError(msg)
        if resolved_target != _CLOUD_TARGET and qc_node_id is None:
            msg = f"qc_node_id is required for target '{resolved_target}'."
            raise ValueError(msg)

        super().__init__(
            resolved_url,
            resolved_api_key,
            n_shots,
            timeout_msec,
            polling_interval=polling_interval,
            polling_timeout_msec=polling_timeout_msec,
            compiler=compiler or OptQCCompiler(),
        )
        self._target = resolved_target
        self._qc_node_id = qc_node_id

        if resolved_target == _CLOUD_TARGET:
            self._backend: _OptQCBackend = _OptQCCloudBackend(self._request)
        else:
            assert qc_node_id is not None  # noqa: S101  # guaranteed by the check above
            self._backend = _OptQCSchedulerBackend(self._request, resolved_target, qc_node_id)

    def health(self) -> bool:
        """Check the health of the API service.

        Returns:
            ``True`` if the service is healthy.
        """
        return self._backend.health()

    def version(self) -> dict[str, str]:
        """Get the service versions.

        Returns:
            dict[str, str]: A dictionary containing version information.
                For cloud targets: ``{"cloud_version": "...", "user_api_version": "..."}``.
                For direct targets:
                ``{"scheduler_version": "...", "submission_api_version": "..."}``.
        """
        return self._backend.version()

    def list_qc_nodes(self) -> list[dict[str, Any]]:
        """List available QC nodes.

        Returns:
            List of QC node information dictionaries.
        """
        return self._backend.list_qc_nodes()

    def submit_job(self, representation: ReprType) -> OptQCSubmissionContext:
        """Submit a job for asynchronous execution.

        The input representation is compiled down to
        :class:`~claveles.machinery.MachineryRepr` before submission,
        preserving intermediate representations for result decoding.

        Args:
            representation: Quantum program in circuit, graph, or machinery representation.

        Returns:
            Submission context for tracking and retrieving the job result.

        Raises:
            TypeError: If ``representation`` is not a ``CircuitRepr``, a ``GraphRepr`` or a
                ``MachineryRepr``. Refused here, before anything is compiled or uploaded, and with the
                same class and wording every other client uses for the same misuse.
        """  # noqa: DOC502  # TypeError comes from `check_representation`
        # `check_representation` exists so that this reads identically whichever client was asked: a
        # client raising a ValueError of its own here would break an `except TypeError` written against
        # the simulator on the way to hardware.
        check_representation(representation)

        circuit_repr: CircuitRepr | None = None
        graph_repr: GraphRepr | None = None
        machinery_repr: MachineryRepr | None = None

        if isinstance(representation, CircuitRepr):
            circuit_repr = representation
            # The conversion refuses a state the machine cannot prepare, naming the modes and what to
            # write instead, so nothing is checked about the initial states here.
            graph_repr = self._compiler.circuit_to_graph(circuit_repr)

        if isinstance(representation, GraphRepr):
            graph_repr = representation

        if graph_repr is not None:
            machinery_repr = self._compiler.graph_to_machinery(graph_repr)

        if isinstance(representation, MachineryRepr):
            machinery_repr = representation

        # Every accepted representation lowers to machinery above, so this only records that.
        assert machinery_repr is not None  # noqa: S101

        job_id = self._backend.submit_job(
            machinery_repr, _JobSettings(n_shots=self.n_shots, timeout_msec=self._timeout_msec)
        )

        return OptQCSubmissionContext(
            job_id=job_id,
            circuit_repr=circuit_repr,
            graph_repr=graph_repr,
            machinery_repr=machinery_repr,
        )

    def get_job_status(self, context: OptQCSubmissionContext) -> JobStatus:
        """Get the current status of a submitted job.

        Args:
            context: Submission context returned by ``submit_job``.

        Returns:
            JobStatus: The state the job is in, under the same spelling
                :meth:`~claveles.client.optqc_common.OptQCClientBase.wait_for_completion` reports.
        """
        return self._backend.get_job_status(context.job_id)

    def get_job_machinery_result(self, job_id: str) -> MachineryResult:
        """Get the raw machinery result of a completed job.

        This is a lower-level method that returns only the machinery-level
        result without decoding to higher representation levels.

        Args:
            job_id: Job identifier.

        Returns:
            Raw machinery result.
        """
        return self._backend.get_job_result(job_id)

    def get_job_result(self, context: OptQCSubmissionContext) -> OptQCClientResult:
        """Get the decoded result of a completed job.

        Decodes the raw machinery result back through the representation
        levels that were used during compilation (machinery → graph → circuit).

        Args:
            context: Submission context returned by ``submit_job``.

        Returns:
            Decoded job result at all available representation levels.

        Raises:
            ClientServerError: If the job did not succeed, or the result cannot be downloaded or
                decoded. Both routes report these the same way.
        """  # noqa: DOC502  # raised by the backend's get_job_result
        decoder = OptQCResultDecoder()
        circuit_result: CircuitResult | None = None
        graph_result: GraphResult | None = None
        machinery_result = self._backend.get_job_result(context.job_id)

        if context.machinery_repr is not None and context.graph_repr is not None:
            graph_result = decoder.machinery_to_graph(machinery_result, context.machinery_repr)

        if graph_result is not None and context.graph_repr is not None and context.circuit_repr is not None:
            circuit_result = decoder.graph_to_circuit(graph_result, context.graph_repr)

        return OptQCClientResult(
            circuit_result=circuit_result,
            graph_result=graph_result,
            machinery_result=machinery_result,
        )
