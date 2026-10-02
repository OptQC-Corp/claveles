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

"""Direct HTTP client for the Cloud Simulator Server (CSS).

This module talks straight to a running CSS instance over its ``/v1/jobs`` REST API
(``docs/api-server/openapi.yaml`` in the ``simulator`` repository is the wire contract). It exists
to verify a CSS deployment end-to-end from the SDK side. The production path (SDK -> Cloud User API
-> simulator-bridge -> CSS, see the cloud simulator requirements spec I-3) goes through the Cloud
User API instead -- that is :mod:`claveles.client._simulator_remote`, exposed as
``SimulatorClient(mode="remote", ...)``. This module is exposed as
``SimulatorClient(mode="simulator-server", ...)`` instead, for talking to a CSS deployment directly
without going through the Cloud User API and a bridge.

Unlike :class:`claveles.client.optqc_common.OptQCClientBase`, the status mapping here keeps 409 (same
``job_id`` resubmitted with different content), 410 (result permanently discarded), and 429 (queue
full, retry after ``Retry-After``) distinguishable instead of collapsing them into one client-error
type, and the polling loop derives its own deadline from ``timeout_msec`` rather than reusing
``OptQCClientBase``'s (whose ``compiler``-centric constructor and ``list_qc_nodes``-shaped interface
don't fit a CSS-direct client).

Everything it raises comes from :mod:`claveles.client.exceptions`, which every other client raises from
too: the three classes below refine :class:`~claveles.client.exceptions.ClientRequestError` with the CSS
statuses that deserve their own name, and a failed job or an expired wait is the same
:class:`~claveles.client.exceptions.JobFailedError` / :class:`~claveles.client.exceptions.JobTimeoutError`
that ``mode="local"`` and :class:`~claveles.client.OptQCClient` raise.

CSS requires a client-supplied idempotency key (``job_id`` + ``lease_id``); a direct SDK caller has
no scheduler-issued lease, so both are generated here with ``uuid4()`` and never surfaced to the
caller.
"""

from __future__ import annotations

import base64
import logging
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any

import requests

from claveles.client._simulator_program import (
    DEFAULT_SERVER_TIMEOUT_MSEC,
    POLLING_TIMEOUT_MARGIN_MSEC,
    TERMINAL_STATUSES,
    decode_result_payload,
    parse_timestamp,
    serialize_program,
)
from claveles.client.exceptions import (
    ClavelesClientError,
    ClientAuthError,
    ClientRequestError,
    ClientServerError,
    JobFailedError,
    JobTimeoutError,
    SimulatorServerBusyError,
    SimulatorServerConflictError,
    SimulatorServerGoneError,
)

if TYPE_CHECKING:
    from datetime import datetime

    from claveles.circuit.program import CircuitRepr
    from claveles.client.abstract import ResultType
    from claveles.client.simulator_client import ShotResult
    from claveles.graph.program import GraphRepr
    from claveles.machinery.program import MachineryRepr
    from claveles.support import StateRepresentationOption

logger = logging.getLogger(__name__)

_SUBMIT_MAX_ATTEMPTS = 3
"""Retry budget for the initial submit POST, whose response (not the request) can get lost."""

_SUBMIT_RETRY_DELAY_SECONDS = 1.0

# Mirrors src/claveles/exception.h's `claveles::error` namespace and
# server/claveles_server/services/error_mapping.py's `_ERROR_ID_TO_CODE`. Keep the three in sync.
_ERROR_CODE_TO_ID: dict[str, int] = {
    "FILE_NOT_FOUND": 101,
    "INVALID_SETTINGS": 201,
    "INVALID_OPERATION": 202,
    "INVALID_STATE": 203,
    "INVALID_FF": 204,
    "OVERFLOW": 205,
    "NAN": 206,
    "TIMEOUT": 207,
    "UNKNOWN": 299,
    "LARGE_INDEX": 301,
    "SMALL_INDEX": 302,
    "KEY_NOT_FOUND": 303,
    "UNREACHABLE": 501,
}


@dataclass(frozen=True)
class SimulatorServerResult:
    """The result of executing a representation on a Cloud Simulator Server."""

    result: ResultType
    """What the submitted program produced, of the type its representation produces.

    One field rather than three optional ones, so "the result of this run" needs no search for the
    populated one and cannot be asked for as a level the run never had."""

    shot_results: list[ShotResult] | None
    """What each shot produced, or ``None`` for graph/machinery, which pair nothing.

    ``None`` rather than an empty list, because "this run has no per-shot states to report" and "this
    run had zero shots" are different answers, and because
    :attr:`~claveles.client.SimulatorClientResult.shot_results` reads it back as the same absence
    ``mode="local"`` produces. CSS always applies the ``first_only`` save policy for circuits, so at
    most shot 0 carries a state."""

    simulator_version: str
    """Version of the simulator binding that executed this job, as reported by CSS."""

    resolved_backend: str
    """The backend CSS actually used, including the result of any automatic selection."""

    effective_seed: int
    """The random seed CSS actually used."""

    queued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None


def _request(url: str, api_key: str, method: str, path: str, **kwargs: Any) -> requests.Response:  # noqa: ANN401
    """Send an HTTP request to a Cloud Simulator Server instance and classify the response.

    Args:
        url: CSS base URL (trailing slashes are stripped).
        api_key: Bearer token accepted by the CSS instance.
        method: HTTP method.
        path: API endpoint path (leading slashes are handled).
        **kwargs: Additional keyword arguments passed to ``requests.request``.

    Returns:
        Response object for successful requests (2xx/3xx).

    Raises:
        ClientServerError: If the server returns a 5xx status code or the connection fails.
        ClientAuthError: If authentication fails (401/403).
        SimulatorServerBusyError: If the server returns 429 (execution queue full).
        SimulatorServerConflictError: If the server returns 409.
        SimulatorServerGoneError: If the server returns 410.
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
    if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        retry_after = response.headers.get("Retry-After")
        msg = f"Cloud Simulator Server execution queue is full: {response.text}"
        raise SimulatorServerBusyError(msg, float(retry_after) if retry_after is not None else None)
    if response.status_code == HTTPStatus.CONFLICT:
        msg = f"Conflict {response.status_code}: {response.text}"
        raise SimulatorServerConflictError(msg)
    if response.status_code == HTTPStatus.GONE:
        msg = f"Gone {response.status_code}: {response.text}"
        raise SimulatorServerGoneError(msg)
    if response.status_code >= HTTPStatus.BAD_REQUEST:
        msg = f"Client error {response.status_code}: {response.text}"
        raise ClientRequestError(msg)

    return response


def _release_job(url: str, api_key: str, job_id: str) -> None:
    """Best-effort ``DELETE`` of a job whose result has been read, so CSS frees the payload at once.

    CSS otherwise holds the payload until ``result_ttl_seconds`` (default 24h) elapses; this is a
    single-worker shared resource, so releasing eagerly is good citizenship. Failures are logged and
    swallowed -- this is a courtesy cleanup, not part of the job's correctness.

    Only a job whose result was collected is released. A job that failed is left on the server for its
    TTL instead: its detail record is the only account of why it failed, and deleting it as part of
    raising made that account unreachable for exactly the run someone would want to look into.
    """
    try:
        _request(url, api_key, "DELETE", f"/v1/jobs/{job_id}")
    except ClavelesClientError:
        logger.debug("Failed to release job %s on the Cloud Simulator Server", job_id, exc_info=True)


def _build_job_body(  # noqa: PLR0913  # one keyword per JobCreateRequest field
    job_id: str,
    lease_id: str,
    program_type: str,
    program: str,
    n_shots: int,
    *,
    backend: str | None,
    seed: int | None,
    timeout_msec: int | None,
    max_bond_dim: int | None = None,
    svd_cutoff: float | None = None,
    track_truncation_error: bool | None = None,
    state_representation: str | None = None,
    squeezing_level: float | None = None,
) -> dict[str, Any]:
    """Build a ``JobCreateRequest`` body, omitting fields the caller left unspecified.

    ``backend`` is omitted (rather than sent as ``null``) when unspecified so the server resolves
    it automatically; ``timeout_msec``/``seed``/``max_bond_dim``/``svd_cutoff``/
    ``track_truncation_error``/``state_representation``/``squeezing_level`` are omitted the same way
    -- the last five are meaningless for at least one program type (circuit-only vs.
    graph/machinery-only, api-design.md S-5), so the caller passes only the ones relevant to
    ``program_type``.

    Returns:
        A JSON-serializable request body.
    """
    body: dict[str, Any] = {
        "job_id": job_id,
        "lease_id": lease_id,
        "program_type": program_type,
        "program": program,
        "n_shots": n_shots,
    }
    if backend is not None:
        body["backend"] = backend
    if timeout_msec is not None:
        body["timeout_msec"] = timeout_msec
    if seed is not None:
        body["seed"] = seed
    if max_bond_dim is not None:
        body["max_bond_dim"] = max_bond_dim
    if svd_cutoff is not None:
        body["svd_cutoff"] = svd_cutoff
    if track_truncation_error is not None:
        body["track_truncation_error"] = track_truncation_error
    if state_representation is not None:
        body["state_representation"] = state_representation
    if squeezing_level is not None:
        body["squeezing_level"] = squeezing_level
    return body


def _poll_until_terminal(
    url: str, api_key: str, job_id: str, timeout_msec: int | None, polling_interval: float
) -> dict[str, Any]:
    """Poll a job's status until it reaches ``SUCCESS``/``FAILURE``/``TIMEOUT``.

    Returns:
        The last ``JobDetailResponse`` body fetched, whose ``status`` is guaranteed terminal.

    Raises:
        JobTimeoutError: If the client-side polling deadline elapses first. Also a ``TimeoutError``,
            and the same class :meth:`~claveles.client.optqc_common.OptQCClientBase.wait_for_completion`
            raises for its own deadline.
    """
    deadline_msec = (
        timeout_msec if timeout_msec is not None else DEFAULT_SERVER_TIMEOUT_MSEC
    ) + POLLING_TIMEOUT_MARGIN_MSEC
    deadline = monotonic() + deadline_msec / 1000
    logger.info("Waiting for job %s on %s (polling timeout: %.1f min)", job_id, url, deadline_msec / 60_000)

    prev_status: str | None = None
    while True:
        if monotonic() >= deadline:
            msg = f"Job {job_id} did not complete within the polling deadline ({deadline_msec} ms)."
            raise JobTimeoutError(msg)

        detail = _request(url, api_key, "GET", f"/v1/jobs/{job_id}").json()
        status = detail["status"]
        if status != prev_status:
            logger.info("Job %s status: %s (queue_position=%s)", job_id, status, detail.get("queue_position"))
            prev_status = status
        if status in TERMINAL_STATUSES:
            return detail
        sleep(polling_interval)


def _submit_job(  # noqa: PLR0913, PLR0917  # one keyword per JobCreateRequest field
    url: str,
    api_key: str,
    job_id: str,
    program_type: str,
    program: str,
    n_shots: int,
    *,
    backend: str | None,
    seed: int | None,
    timeout_msec: int | None,
    resource_squeezing_level: float,
    max_bond_dim: int | None,
    svd_cutoff: float,
    track_truncation_error: bool,
    state_representation: StateRepresentationOption,
) -> None:
    """POST a ``JobCreateRequest``, sending only the fields meaningful for ``program_type``."""
    is_circuit = program_type == "CIRCUIT"
    body = _build_job_body(
        job_id,
        str(uuid.uuid4()),
        program_type,
        program,
        n_shots,
        backend=backend,
        seed=seed,
        timeout_msec=timeout_msec,
        max_bond_dim=max_bond_dim if is_circuit else None,
        svd_cutoff=svd_cutoff if is_circuit else None,
        track_truncation_error=track_truncation_error if is_circuit else None,
        state_representation=state_representation if is_circuit else None,
        squeezing_level=None if is_circuit else resource_squeezing_level,
    )
    _request(url, api_key, "POST", "/v1/jobs", json=body)


def run_representation_on_simulator_server(  # noqa: PLR0913  # mirrors the public simulator configuration options
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
) -> SimulatorServerResult:
    """Submit a representation to a Cloud Simulator Server, poll until terminal, decode the result.

    Args:
        url: CSS base URL.
        api_key: Bearer token accepted by the CSS instance.
        representation: Circuit, graph, or machinery representation to execute.
        n_shots: Number of measurement shots.
        backend: Explicit backend name, or ``None`` to let CSS resolve it automatically.
        seed: Explicit random seed, or ``None`` to let CSS apply its default.
        timeout_msec: Server-side execution timeout in milliseconds, or ``None`` to let CSS apply
            its own default. Also used to size the client-side polling deadline.
        resource_squeezing_level: Squeezing level of the resource state. Sent as
            ``squeezing_level``; required by CSS for graph/machinery, ignored for circuits.
        max_bond_dim: Maximum MPS bond dimension for the Fock simulator, or ``None`` for unbounded.
            Circuit-only.
        svd_cutoff: SVD truncation cutoff for the Fock simulator. Circuit-only.
        track_truncation_error: Whether to record the per-shot accumulated SVD truncation error.
            Circuit-only.
        decompose_std: Whether to lower std operations to their intrinsic decomposition before
            serialization. Circuit-only.
        state_representation: Which representation the circuit should be run in. Circuit-only.
        polling_interval: Interval in seconds between job status polls.

    Returns:
        The decoded execution result.

    Raises:
        JobFailedError: If the job reaches ``FAILURE`` or ``TIMEOUT``. Carries
            ``args == (message, error_id)``, the same shape the local simulator binding raises
            directly, so callers can handle both execution modes with the same ``except`` clause.
        JobTimeoutError: If the client-side polling deadline elapses before the job reaches a terminal
            state. Also a ``TimeoutError``.
        ClientServerError: If submitting the job keeps failing after ``_SUBMIT_MAX_ATTEMPTS``
            retries.
    """  # noqa: DOC502  # JobTimeoutError propagates from _poll_until_terminal
    program_type, program_bytes = serialize_program(
        representation, decompose_std=decompose_std, state_representation=state_representation
    )
    program = base64.b64encode(program_bytes).decode("ascii")
    job_id = str(uuid.uuid4())
    # A lost connection here leaves the caller unable to tell whether the POST ever reached CSS.
    # Retrying under the same job_id is safe either way: CSS's idempotency rule (api-design.md
    # §5.1/§5.2) treats a same-job_id resubmission with identical content as either an idempotent
    # resend (if the first attempt did land) or a fresh accept (if it didn't) -- never a conflict,
    # since only the (per-attempt) lease_id differs. Without this, a lost response would leave an
    # orphaned job on CSS until its TTL expires instead.
    for attempt in range(1, _SUBMIT_MAX_ATTEMPTS + 1):
        try:
            _submit_job(
                url,
                api_key,
                job_id,
                program_type,
                program,
                n_shots,
                backend=backend,
                seed=seed,
                timeout_msec=timeout_msec,
                resource_squeezing_level=resource_squeezing_level,
                max_bond_dim=max_bond_dim,
                svd_cutoff=svd_cutoff,
                track_truncation_error=track_truncation_error,
                state_representation=state_representation,
            )
            break
        except ClientServerError:
            if attempt == _SUBMIT_MAX_ATTEMPTS:
                raise
            logger.warning("Failed to submit job %s (attempt %d/%d); retrying.", job_id, attempt, _SUBMIT_MAX_ATTEMPTS)
            sleep(_SUBMIT_RETRY_DELAY_SECONDS)

    detail = _poll_until_terminal(url, api_key, job_id, timeout_msec, polling_interval)
    status = detail["status"]

    if status != "SUCCESS":
        error = detail.get("error") or {}
        message = error.get("message") or f"Job {job_id} failed with status {status}."
        error_id = _ERROR_CODE_TO_ID.get(error.get("code", "UNKNOWN"), _ERROR_CODE_TO_ID["UNKNOWN"])
        # The job is deliberately not released here: see `_release_job`.
        raise JobFailedError(message, error_id)

    envelope = _request(url, api_key, "GET", f"/v1/jobs/{job_id}/result").json()
    result, shot_results = decode_result_payload(program_type, base64.b64decode(envelope["payload"]), n_shots)
    _release_job(url, api_key, job_id)

    timeline = detail.get("timeline") or {}
    return SimulatorServerResult(
        result=result,
        shot_results=shot_results,
        simulator_version=envelope.get("simulator_version", ""),
        resolved_backend=detail["resolved_backend"],
        effective_seed=detail["effective_seed"],
        queued_at=parse_timestamp(timeline.get("queued_at")),
        started_at=parse_timestamp(timeline.get("started_at")),
        finished_at=parse_timestamp(timeline.get("finished_at")),
    )
