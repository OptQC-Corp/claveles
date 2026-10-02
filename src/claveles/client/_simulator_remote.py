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

"""Cloud User API client for the production remote path to a Cloud Simulator Server (CSS).

This module reaches CSS the way production traffic does: SDK -> Cloud User API
(``/api/user/v1/simulator/jobs...``) -> ``simulator-bridge`` -> CSS. It is exposed as
``SimulatorClient(mode="remote", ...)``. :mod:`claveles.client._simulator_server` reaches the same CSS
directly instead, exposed as ``SimulatorClient(mode="simulator-server", ...)`` -- a
verification/development path that skips the Cloud User API and the bridge entirely.

Submission is the same three-step S3 flow :class:`~claveles.client.optqc_client.OptQCClient` uses for
hardware jobs (Step 1: get a job id and a presigned upload URL; Step 2: PUT the program to S3; Step 3:
register the job), except the uploaded object is a raw serialized protobuf (``application/octet-stream``)
rather than JSON, and Step 3's ``settings`` follow ``SimulatorJobExecutionSettings``
(``cloud/src/optqc_cloud_api/app/models/schemas.py``), not ``JobExecutionSettings``.

Two things are missing from this path relative to talking to CSS directly, both because
``SimulatorJobExecutionSettings`` does not carry them (unlike CSS's own ``JobCreateRequest``):

- ``state_representation``: with no wire field to carry an explicit choice through the bridge, this
  client accepts only the default (``"auto"``) for a circuit run and refuses any other value, rather
  than silently letting CSS's own auto-resolution override a choice it was never told about.
- ``resolved_backend`` / ``effective_seed``: ``SimulatorJobExecutionReport`` reports neither, so
  :class:`RemoteSimulatorResult` does not carry them either (contrast
  :class:`~claveles.client._simulator_server.SimulatorServerResult`, which does).

The result envelope also arrives differently: CSS's ``ResultEnvelope`` JSON, gzip-compressed by CSS
itself, is relayed byte-for-byte by ``simulator-bridge`` and by the (Cloud <-> Scheduler) bridge all
the way into the S3 object the ``download_url`` points at (see
``simulator-bridge/src/simulator_bridge/agent.py::_upload_result`` and
``bridge/src/bridge/bridge_agent.py::_fetch_and_upload_simulator_result``, both of which read gzip
bytes and forward them, and both of which set ``Content-Encoding: gzip`` on the way out so S3 stores
that as object metadata). A GET of the presigned ``download_url`` may or may not come back already
decompressed depending on whether the HTTP client along the way honoured that header, so this module
checks for the gzip magic bytes and only decompresses when they are present, the same defensive check
``simulator-bridge`` itself uses.
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any

import requests

from claveles.client._simulator_program import (
    DEFAULT_SERVER_TIMEOUT_MSEC,
    POLLING_TIMEOUT_MARGIN_MSEC,
    TERMINAL_STATUSES,
    decode_result_payload,
    serialize_program,
)
from claveles.client.exceptions import (
    ClientAuthError,
    ClientRequestError,
    ClientServerError,
    JobFailedError,
    JobTimeoutError,
)

if TYPE_CHECKING:
    from claveles.circuit.program import CircuitRepr
    from claveles.client.abstract import ResultType
    from claveles.client.simulator_client import ShotResult
    from claveles.graph.program import GraphRepr
    from claveles.machinery.program import MachineryRepr
    from claveles.support import StateRepresentationOption

logger = logging.getLogger(__name__)

_CREATE_PATH = "api/user/v1/simulator/jobs"


@dataclass(frozen=True)
class RemoteSimulatorResult:
    """The result of executing a representation through the Cloud User API.

    Deliberately narrower than :class:`~claveles.client._simulator_server.SimulatorServerResult`: see the
    module docstring for what the Cloud User API does not report.
    """

    result: ResultType
    """What the submitted program produced. See
    :attr:`~claveles.client._simulator_server.SimulatorServerResult.result`, which this mirrors."""

    shot_results: list[ShotResult] | None
    """What each shot produced, or ``None`` for graph/machinery. See
    :attr:`~claveles.client._simulator_server.SimulatorServerResult.shot_results`, which this mirrors."""

    simulator_version: str
    """Version of the simulator binding that executed this job, as reported by CSS and relayed
    through ``SimulatorJobExecutionReport.simulator_version``."""

    queued_at: datetime | None
    """When the job entered CSS's queue, approximated: see :func:`_derive_timeline`."""

    started_at: datetime | None
    """When CSS started executing the job, approximated: see :func:`_derive_timeline`."""

    finished_at: datetime | None
    """When this client observed the job as finished (client-side wall clock, not CSS's)."""


def _request(url: str, api_key: str, method: str, path: str, **kwargs: Any) -> requests.Response:  # noqa: ANN401
    """Send an HTTP request to the Cloud User API and classify the response.

    Args:
        url: Cloud User API base URL (trailing slashes are stripped).
        api_key: Bearer token accepted by the Cloud User API (the same one
            ``OptQCClient(target="cloud")`` uses).
        method: HTTP method.
        path: API endpoint path (leading slashes are handled).
        **kwargs: Additional keyword arguments passed to ``requests.request``.

    Returns:
        Response object for successful requests (2xx/3xx).

    Raises:
        ClientServerError: If the server returns a 5xx status code or the connection fails.
        ClientAuthError: If authentication fails (401/403).
        ClientRequestError: If the server returns any other 4xx status code.
    """
    full_url = f"{url.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {api_key}"}
    kwargs.setdefault("timeout", 30)

    try:
        response = requests.request(method, full_url, headers=headers, **kwargs)  # noqa: S113
    except requests.RequestException as e:
        msg = f"Failed to connect to {full_url}: {e}"
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


def _upload_program(upload_url: str, program_bytes: bytes) -> None:
    """PUT the raw serialized program proto to the Step 1 presigned S3 URL.

    Unlike the JSON envelope ``mode="simulator-server"`` builds (``program`` base64-encoded inside a
    ``JobCreateRequest``), the S3 object here (``simulator_jobs/{job_id}/program.pb``) is the raw
    proto bytes, matching the ``content_type="application/octet-stream"`` the presigned URL was
    signed with (``generate_simulator_job_id`` in ``cloud/src/optqc_cloud_api/app/apis/user.py``).

    Args:
        upload_url: Presigned S3 PUT URL from Step 1.
        program_bytes: Raw serialized proto bytes.

    Raises:
        ClientServerError: If the upload fails.
    """
    headers = {
        "Content-Type": "application/octet-stream",
        "Host": urllib.parse.urlparse(upload_url).netloc,
    }
    try:
        response = requests.put(upload_url, data=program_bytes, headers=headers, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        msg = f"Failed to upload program to S3: {e}"
        raise ClientServerError(msg) from e


def _build_settings_body(  # noqa: PLR0913  # one keyword per SimulatorJobExecutionSettings field
    n_shots: int,
    *,
    backend: str | None,
    seed: int | None,
    timeout_msec: int | None,
    max_bond_dim: int | None,
    svd_cutoff: float | None,
    track_truncation_error: bool | None,
    resource_squeezing_level: float | None,
) -> dict[str, Any]:
    """Build a ``SimulatorJobExecutionSettings`` body, omitting fields the caller left unspecified.

    Field names follow the Cloud User API schema exactly, which for one field differs from CSS's own
    ``JobCreateRequest``: this is ``resource_squeezing_level``, not CSS's ``squeezing_level``.

    Returns:
        A JSON-serializable ``settings`` object.
    """
    settings: dict[str, Any] = {"n_shots": n_shots}
    if backend is not None:
        settings["backend"] = backend
    if seed is not None:
        settings["seed"] = seed
    if timeout_msec is not None:
        settings["timeout_msec"] = timeout_msec
    if resource_squeezing_level is not None:
        settings["resource_squeezing_level"] = resource_squeezing_level
    if max_bond_dim is not None:
        settings["max_bond_dim"] = max_bond_dim
    if svd_cutoff is not None:
        settings["svd_cutoff"] = svd_cutoff
    if track_truncation_error is not None:
        settings["track_truncation_error"] = track_truncation_error
    return settings


def _poll_until_terminal(
    url: str, api_key: str, job_id: str, timeout_msec: int | None, polling_interval: float
) -> dict[str, Any]:
    """Poll a simulator job's status until it reaches ``SUCCESS``/``FAILURE``/``TIMEOUT``.

    The Cloud User API additionally reports a non-terminal ``SUBMITTED`` phase CSS itself never
    reports (waiting for ``simulator-bridge`` to pick the job up); it is simply one more status this
    loop keeps polling through.

    Returns:
        The last ``SimulatorJobResponse`` body fetched, whose ``status`` is guaranteed terminal.

    Raises:
        JobTimeoutError: If the client-side polling deadline elapses first. Also a ``TimeoutError``.
    """
    deadline_msec = (
        timeout_msec if timeout_msec is not None else DEFAULT_SERVER_TIMEOUT_MSEC
    ) + POLLING_TIMEOUT_MARGIN_MSEC
    deadline = monotonic() + deadline_msec / 1000
    logger.info("Waiting for job %s via %s (polling timeout: %.1f min)", job_id, url, deadline_msec / 60_000)

    prev_status: str | None = None
    while True:
        if monotonic() >= deadline:
            msg = f"Job {job_id} did not complete within the polling deadline ({deadline_msec} ms)."
            raise JobTimeoutError(msg)

        detail = _request(url, api_key, "GET", f"{_CREATE_PATH}/{job_id}").json()
        status = detail["status"]
        if status != prev_status:
            logger.info("Job %s status: %s", job_id, status)
            prev_status = status
        if status in TERMINAL_STATUSES:
            return detail
        sleep(polling_interval)


_GZIP_MAGIC = b"\x1f\x8b"
"""First two bytes of a gzip stream, per RFC 1952 §2.3.1."""


def _download_result_envelope(download_url: str) -> dict[str, Any]:
    """GET the presigned result URL and return the decoded ``ResultEnvelope`` JSON.

    The S3 object is the gzip-compressed bytes CSS itself produced, stored with
    ``Content-Encoding: gzip`` (see the module docstring). A plain HTTP client following that header
    decompresses transparently; this checks for the gzip magic bytes first and only decompresses
    itself when they are still present, so it works whichever way the presigned GET actually behaves.

    Args:
        download_url: Presigned S3 GET URL from the terminal job response.

    Returns:
        The decoded ``ResultEnvelope`` (``program_type``, ``simulator_version``, base64 ``payload``).

    Raises:
        ClientServerError: If the download fails, or the body is not the envelope it claims to be.
    """
    try:
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        msg = f"Failed to download result: {e}"
        raise ClientServerError(msg) from e

    raw = response.content
    if raw[: len(_GZIP_MAGIC)] == _GZIP_MAGIC:
        raw = gzip.decompress(raw)

    try:
        envelope = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        msg = f"Failed to decode result envelope: {e}"
        raise ClientServerError(msg) from e
    if not isinstance(envelope, dict) or "payload" not in envelope:
        msg = f"Result envelope is missing 'payload': {envelope!r}"
        raise ClientServerError(msg)
    return envelope


def _derive_timeline(
    execution_report: dict[str, Any], finished_at: datetime
) -> tuple[datetime | None, datetime | None]:
    """Approximate CSS's absolute queue/execution timestamps from the durations Cloud reports.

    ``SimulatorJobExecutionReport`` carries only ``queue_time_msec``/``execution_time_msec``
    (durations), unlike CSS's own ``JobDetailResponse.timeline``, which carries absolute
    ``queued_at``/``started_at``/``finished_at``. Anchoring the two durations to the client-observed
    ``finished_at`` keeps :attr:`~claveles.client.SimulatorClientResult.wait_time` and
    :attr:`~claveles.client.SimulatorClientResult.execution_time` exactly equal to what Cloud reported;
    only the absolute placement on the clock is an approximation.

    Args:
        execution_report: The terminal response's ``execution_report`` object (or ``{}``).
        finished_at: When this client observed the job as finished.

    Returns:
        A ``(queued_at, started_at)`` tuple, either or both ``None`` if the corresponding duration
        was not reported.
    """
    execution_time_msec = execution_report.get("execution_time_msec")
    started_at = finished_at - timedelta(milliseconds=execution_time_msec) if execution_time_msec is not None else None

    queue_time_msec = execution_report.get("queue_time_msec")
    queued_at = (
        started_at - timedelta(milliseconds=queue_time_msec)
        if started_at is not None and queue_time_msec is not None
        else None
    )
    return queued_at, started_at


def _submit_program(url: str, api_key: str, program_bytes: bytes) -> str:
    """Run Steps 1 and 2: get a job id and presigned upload URL, then upload the program.

    Args:
        url: Cloud User API base URL.
        api_key: Bearer token accepted by the Cloud User API.
        program_bytes: Raw serialized proto bytes.

    Returns:
        The job id Step 1 generated.
    """
    create_response = _request(url, api_key, "POST", _CREATE_PATH).json()
    job_id: str = create_response["job_id"]
    _upload_program(create_response["upload_url"], program_bytes)
    return job_id


def _register_job(  # noqa: PLR0913  # one keyword per SimulatorJobExecutionSettings field
    url: str,
    api_key: str,
    job_id: str,
    program_type: str,
    n_shots: int,
    *,
    backend: str | None,
    seed: int | None,
    timeout_msec: int | None,
    resource_squeezing_level: float,
    max_bond_dim: int | None,
    svd_cutoff: float,
    track_truncation_error: bool,
) -> None:
    """Run Step 3: register the job with its execution settings, once its program is on S3."""
    is_circuit = program_type == "CIRCUIT"
    settings = _build_settings_body(
        n_shots,
        backend=backend,
        seed=seed,
        timeout_msec=timeout_msec,
        max_bond_dim=max_bond_dim if is_circuit else None,
        svd_cutoff=svd_cutoff if is_circuit else None,
        track_truncation_error=track_truncation_error if is_circuit else None,
        resource_squeezing_level=None if is_circuit else resource_squeezing_level,
    )
    _request(
        url,
        api_key,
        "POST",
        f"{_CREATE_PATH}/{job_id}/submit",
        json={"settings": settings, "program_type": program_type},
    )


def _build_result(job_id: str, program_type: str, n_shots: int, detail: dict[str, Any]) -> RemoteSimulatorResult:
    """Turn a terminal ``SimulatorJobResponse`` into a :class:`RemoteSimulatorResult`.

    Args:
        job_id: Job id, used only to name it in a raised error.
        program_type: ``"CIRCUIT"``/``"GRAPH"``/``"MACHINERY"``, as returned by
            :func:`~claveles.client._simulator_program.serialize_program`.
        n_shots: Number of measurement shots that were requested.
        detail: The terminal ``SimulatorJobResponse`` body :func:`_poll_until_terminal` returned.

    Returns:
        The decoded execution result.

    Raises:
        JobFailedError: If ``detail["status"]`` is not ``"SUCCESS"``.
        ClientServerError: If the response is missing ``download_url``, or the result download or
            decode fails.
    """
    status = detail["status"]
    execution_report = detail.get("execution_report") or {}
    if status != "SUCCESS":
        message = execution_report.get("error_message") or f"Job {job_id} failed with status {status}."
        raise JobFailedError(message)

    download_url = detail.get("download_url")
    if not download_url:
        msg = f"Job {job_id} succeeded but no download_url was returned."
        raise ClientServerError(msg)

    envelope = _download_result_envelope(download_url)
    result, shot_results = decode_result_payload(program_type, base64.b64decode(envelope["payload"]), n_shots)

    finished_at = datetime.now(tz=timezone.utc)  # noqa: UP017  # this package supports 3.10; datetime.UTC is 3.11+
    queued_at, started_at = _derive_timeline(execution_report, finished_at)
    return RemoteSimulatorResult(
        result=result,
        shot_results=shot_results,
        simulator_version=envelope.get("simulator_version", ""),
        queued_at=queued_at,
        started_at=started_at,
        finished_at=finished_at,
    )


def run_representation_remotely(  # noqa: PLR0913  # mirrors the public simulator configuration options
    url: str,
    api_key: str,
    representation: CircuitRepr | GraphRepr | MachineryRepr,
    n_shots: int,
    *,
    backend: str | None,
    seed: int | None,
    timeout_msec: int | None,
    resource_squeezing_level: float,
    max_bond_dim: int | None,
    svd_cutoff: float,
    track_truncation_error: bool,
    decompose_std: bool,
    state_representation: StateRepresentationOption,
    polling_interval: float,
) -> RemoteSimulatorResult:
    """Submit a representation through the Cloud User API, poll until terminal, decode the result.

    Args:
        url: Cloud User API base URL.
        api_key: Bearer token accepted by the Cloud User API.
        representation: Circuit, graph, or machinery representation to execute.
        n_shots: Number of measurement shots.
        backend: Explicit backend name, or ``None`` to let CSS resolve it automatically.
        seed: Explicit random seed, or ``None`` to let CSS apply its default.
        timeout_msec: Server-side execution timeout in milliseconds, or ``None`` to let CSS apply
            its own default. Also used to size the client-side polling deadline.
        resource_squeezing_level: Squeezing level of the resource state. Required by CSS for
            graph/machinery, ignored for circuits.
        max_bond_dim: Maximum MPS bond dimension for the Fock simulator, or ``None`` for unbounded.
            Circuit-only.
        svd_cutoff: SVD truncation cutoff for the Fock simulator. Circuit-only.
        track_truncation_error: Whether to record the per-shot accumulated SVD truncation error.
            Circuit-only.
        decompose_std: Whether to lower std operations to their intrinsic decomposition before
            serialization. Circuit-only.
        state_representation: Which representation the circuit should be run in. Circuit-only.
            Must be ``"auto"``: see the module docstring for why this path cannot honour another
            choice. Callers reach this only via ``SimulatorClient``, which refuses any other value
            before calling here.
        polling_interval: Interval in seconds between job status polls.

    Returns:
        The decoded execution result.

    Raises:
        JobFailedError: If the job reaches ``FAILURE`` or ``TIMEOUT``.
        JobTimeoutError: If the client-side polling deadline elapses before the job reaches a
            terminal state. Also a ``TimeoutError``.
        ClientServerError: If a step of the submission, polling, or result download fails.
    """  # noqa: DOC502  # JobTimeoutError propagates from _poll_until_terminal
    program_type, program_bytes = serialize_program(
        representation, decompose_std=decompose_std, state_representation=state_representation
    )
    job_id = _submit_program(url, api_key, program_bytes)
    _register_job(
        url,
        api_key,
        job_id,
        program_type,
        n_shots,
        backend=backend,
        seed=seed,
        timeout_msec=timeout_msec,
        resource_squeezing_level=resource_squeezing_level,
        max_bond_dim=max_bond_dim,
        svd_cutoff=svd_cutoff,
        track_truncation_error=track_truncation_error,
    )
    detail = _poll_until_terminal(url, api_key, job_id, timeout_msec, polling_interval)
    return _build_result(job_id, program_type, n_shots, detail)
