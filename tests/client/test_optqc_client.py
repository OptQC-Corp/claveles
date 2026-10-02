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

"""Tests for OptQCClient (unified client with mock HTTP)."""

import json
import logging
from collections.abc import Callable
from http import HTTPStatus
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
import requests

from claveles.circuit import BosonicState, CircuitRepr, FockState
from claveles.circuit.ops import intrinsic
from claveles.circuit.state import HardwareConstrainedSqueezedState
from claveles.client import (
    ClientServerError,
    JobTimeoutError,
    OptQCAuthError,
    OptQCClient,
    OptQCClientError,
    OptQCClientResult,
    OptQCCompiler,
    OptQCJobError,
    OptQCServerError,
    OptQCSubmissionContext,
)

# --- Dummy Classes for isinstance checks ---


class DummyCircuitRepr:
    """Dummy class for circuit representation."""


class DummyGraphRepr:
    """Dummy class for graph representation."""


class DummyMachineryRepr:
    """Dummy class for machinery representation."""

    n_local_macronodes: int = 129
    n_steps: int = 1
    readout_macronode_indices: ClassVar[list[int]] = [0]


def patched_machinery_repr(test: Callable) -> Callable:
    """Make ``DummyMachineryRepr`` pass for a ``MachineryRepr`` everywhere the client looks for one.

    Two places, which is why this is one decorator rather than a ``patch`` at each test: the check that
    refuses anything that is none of the three representations (``abstract.representation_kind``, shared
    with every other client) and ``submit_job``'s own branch on which level to lower from.
    """
    for target in ("claveles.client.abstract.MachineryRepr", "claveles.client.optqc_client.MachineryRepr"):
        test = patch(target, DummyMachineryRepr)(test)
    return test


# --- Helpers ---


def _make_response(status_code: int = 200, json_data: dict[str, Any] | None = None, content: bytes = b"") -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.content = content
    resp.raise_for_status.side_effect = requests.HTTPError(response=resp) if status_code >= 400 else None
    return resp


def _make_result_content() -> bytes:
    """Create machinery result JSON content.

    ``requests`` transparently gzip-decodes the response body based on the
    ``Content-Encoding`` header, so ``response.content`` already contains
    plain JSON bytes by the time application code sees it.
    """
    result_data = {
        # Hardware returns the singular key (see json_to_machinery_result).
        "readout_macronode_indices": [0],
        "shots": [
            {"values": [1.0, 2.0, 3.0, 4.0]},
        ],
    }
    return json.dumps(result_data).encode()


def _requested_url(mock_request: MagicMock) -> str:
    """Extract the URL passed to the mocked ``requests.request`` call."""
    call_args = mock_request.call_args
    return call_args[0][1]


@pytest.fixture
def direct_client() -> OptQCClient:
    """Fixture for direct backend client."""
    return OptQCClient(
        url="https://scheduler.local",
        api_key="test-api-key",
        target="direct",
        qc_node_id="node-001",
        n_shots=100,
        timeout_msec=60000,
    )


@pytest.fixture
def cloud_client() -> OptQCClient:
    """Fixture for cloud backend client."""
    return OptQCClient(
        url="https://compute.cloud.optqc.com",
        api_key="cloud-api-key",
        target="cloud",
        n_shots=500,
        timeout_msec=120000,
        polling_interval=0.01,
    )


# --- Construction ---


class TestOptQCClientConstruction:
    """Test client construction and validation."""

    def test_api_key_is_accepted(self) -> None:
        """Test that api_key is accepted."""
        client = OptQCClient(
            url="https://compute.cloud.optqc.com",
            api_key="api-key",
            target="cloud",
        )
        assert client._target == "cloud"  # noqa: SLF001

    def test_direct_requires_qc_node_id(self) -> None:
        """Test that direct backend requires qc_node_id."""
        with pytest.raises(ValueError, match="qc_node_id is required"):
            OptQCClient(
                url="https://scheduler.local",
                api_key="api-key",
                target="direct",
            )

    def test_cloud_does_not_require_qc_node_id(self) -> None:
        """Test that cloud backend does not require qc_node_id."""
        client = OptQCClient(
            url="https://compute.cloud.optqc.com",
            api_key="api-key",
            target="cloud",
        )
        assert client._target == "cloud"  # noqa: SLF001

    def test_default_compiler(self) -> None:
        """Test default compiler initialization."""
        client = OptQCClient(
            url="https://scheduler.local",
            api_key="api-key",
            target="direct",
            qc_node_id="node-001",
        )
        assert isinstance(client._compiler, OptQCCompiler)  # noqa: SLF001

    def test_custom_compiler(self) -> None:
        """Test custom compiler initialization."""
        compiler = OptQCCompiler(n_local_macronodes=64)
        client = OptQCClient(
            url="https://scheduler.local",
            api_key="api-key",
            target="direct",
            qc_node_id="node-001",
            compiler=compiler,
        )
        assert client._compiler.n_local_macronodes == 64  # noqa: SLF001

    def test_url_trailing_slash_stripped(self) -> None:
        """Test that trailing slash in URL is stripped."""
        client = OptQCClient(
            url="https://scheduler.local/",
            api_key="api-key",
            target="cloud",
        )
        assert client._url == "https://scheduler.local"  # noqa: SLF001

    def test_unknown_target_raises(self) -> None:
        """Test that an unrecognized target raises ValueError."""
        with pytest.raises(ValueError, match="Unknown target 'bogus'"):
            OptQCClient(
                url="https://scheduler.local",
                api_key="api-key",
                target="bogus",  # ty: ignore[invalid-argument-type]
                qc_node_id="node-001",
            )


# --- Direct Backend ---


class TestDirectBackendHealth:
    """Test health check for direct backend."""

    @patch("claveles.client.optqc_common.requests.request")
    def test_health_ok(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test health check success."""
        mock_request.return_value = _make_response(HTTPStatus.OK)
        assert direct_client.health() is True
        # target="direct" maps to the "direct" URL path segment.
        assert _requested_url(mock_request) == "https://scheduler.local/v1/direct/healthz"

    @patch("claveles.client.optqc_common.requests.request")
    def test_health_server_error(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test health check server error."""
        mock_request.return_value = _make_response(HTTPStatus.INTERNAL_SERVER_ERROR)
        with pytest.raises(OptQCServerError):
            direct_client.health()


class TestRemovedInternalTarget:
    """Test that the former target="internal" is no longer accepted."""

    def test_internal_target_rejected(self) -> None:
        """Test that target="internal" raises ValueError as an unknown target."""
        with pytest.raises(ValueError, match=r"Unknown target 'internal'"):
            OptQCClient(
                url="https://scheduler.local",
                api_key="test-api-key",
                target="internal",  # ty: ignore[invalid-argument-type]
                qc_node_id="node-001",
            )


class TestDirectBackendVersion:
    """Test version retrieval for direct backend."""

    @patch("claveles.client.optqc_common.requests.request")
    def test_version(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test version retrieval."""
        mock_request.return_value = _make_response(
            HTTPStatus.OK,
            json_data={"version": "1.2.3", "api": "2026-01-01"},
        )
        assert direct_client.version() == {
            "scheduler_version": "1.2.3",
            "submission_api_version": "2026-01-01",
        }
        assert _requested_url(mock_request) == "https://scheduler.local/v1/direct/version"


class TestCloudBackendVersion:
    """Test version retrieval for cloud backend."""

    @patch("claveles.client.optqc_common.requests.request")
    def test_version(self, mock_request: MagicMock, cloud_client: OptQCClient) -> None:
        """Test version retrieval."""
        mock_request.return_value = _make_response(
            HTTPStatus.OK,
            json_data={"version": "1.2.3", "api": "2026-01-01"},
        )
        assert cloud_client.version() == {
            "cloud_version": "1.2.3",
            "user_api_version": "2026-01-01",
        }


class TestDirectBackendListQCNodes:
    """Test list_qc_nodes endpoint for direct backend."""

    @patch("claveles.client.optqc_common.requests.request")
    def test_list_qc_nodes(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test listing QC nodes."""
        nodes = [{"qc_node_id": "n1", "status": "ONLINE"}]
        mock_request.return_value = _make_response(HTTPStatus.OK, json_data=nodes)  # ty: ignore[invalid-argument-type]
        result = direct_client.list_qc_nodes()
        assert result == nodes
        assert _requested_url(mock_request) == "https://scheduler.local/v1/direct/qc_nodes"


class TestDirectBackendSubmitJob:
    """Test job submission for direct backend."""

    @patched_machinery_repr
    @patch("claveles.client.optqc_common.requests.request")
    def test_submit_machinery_repr(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test submission of MachineryRepr."""
        mock_request.return_value = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "job-001"})

        machinery = DummyMachineryRepr()
        machinery.n_local_macronodes = 2
        machinery.n_steps = 1

        with patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={"test": True}):
            context = direct_client.submit_job(machinery)  # ty: ignore[invalid-argument-type]

        assert isinstance(context, OptQCSubmissionContext)
        assert context.job_id == "job-001"
        assert _requested_url(mock_request) == "https://scheduler.local/v1/direct/jobs"

    @patched_machinery_repr
    @patch("claveles.client.optqc_common.requests.request")
    def test_submit_uses_timeout_msec(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test that submission uses configured timeout."""
        mock_request.return_value = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "job-002"})

        machinery = DummyMachineryRepr()

        with patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}):
            direct_client.submit_job(machinery)  # ty: ignore[invalid-argument-type]

        call_args = mock_request.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0]
        assert payload["settings"]["timeout_msec"] == 60000
        assert payload["settings"]["n_shots"] == 100


class TestInitialStateRefusal:
    """Test what submission does with a state the machine cannot prepare."""

    @patch("claveles.client.optqc_common.requests.request")
    def test_refuses_non_hcss_initial_states(
        self,
        mock_request: MagicMock,
        direct_client: OptQCClient,
    ) -> None:
        """A circuit carrying such states is refused before anything is submitted, listing the modes."""
        mock_request.return_value = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "job-001"})

        circuit = CircuitRepr("refusal-test")
        circuit.Q(0) | BosonicState.vacuum()
        circuit.Q(2) | FockState.vacuum(cutoff_dim=4)
        for mode in range(3):
            circuit.Q(mode) | intrinsic.PhaseRotation(0.0)

        with pytest.raises(ValueError, match=r"\[0, 2\].*remove"):
            direct_client.submit_job(circuit)

        mock_request.assert_not_called()

    @patch("claveles.client.optqc_common.requests.request")
    def test_a_circuit_with_open_modes_is_what_gets_submitted(
        self,
        mock_request: MagicMock,
        direct_client: OptQCClient,
    ) -> None:
        """Leaving the modes open is what asks for the state the machine supplies, and what converts."""
        mock_request.return_value = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "job-001"})

        circuit = CircuitRepr("refusal-test")
        circuit.Q(0) | intrinsic.PhaseRotation(0.0)

        context = direct_client.submit_job(circuit)

        assert context.job_id == "job-001"

    @patch("claveles.client.optqc_common.requests.request")
    def test_accepts_all_hcss(
        self,
        mock_request: MagicMock,
        direct_client: OptQCClient,
    ) -> None:
        """A circuit whose modes are all HCSS (including non-default phi) converts as placed."""
        mock_request.return_value = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "job-001"})

        circuit = CircuitRepr("refusal-test")
        circuit.Q(1) | HardwareConstrainedSqueezedState(phi=1.0)
        for mode in range(2):
            circuit.Q(mode) | intrinsic.PhaseRotation(0.0)

        direct_client.submit_job(circuit)

    @patched_machinery_repr
    @patch("claveles.client.optqc_common.requests.request")
    def test_a_machinery_repr_carries_no_states_to_refuse(
        self,
        mock_request: MagicMock,
        direct_client: OptQCClient,
    ) -> None:
        """Submitting a MachineryRepr directly goes nowhere near a circuit's initial states."""
        mock_request.return_value = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "job-001"})

        machinery = DummyMachineryRepr()
        with patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}):
            direct_client.submit_job(machinery)  # ty: ignore[invalid-argument-type]


class TestDirectBackendGetJobStatus:
    """Test job status retrieval for direct backend."""

    @patch("claveles.client.optqc_common.requests.request")
    def test_get_status(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test status retrieval."""
        mock_request.return_value = _make_response(HTTPStatus.OK, json_data={"status": "RUNNING"})
        context = OptQCSubmissionContext(job_id="job-001")
        assert direct_client.get_job_status(context) == "RUNNING"
        assert _requested_url(mock_request) == "https://scheduler.local/v1/direct/jobs/job-001"


class TestDirectBackendGetJobResult:
    """Test job result retrieval for direct backend."""

    @patch("claveles.client.optqc_common.requests.request")
    def test_get_result(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test result retrieval."""
        mock_request.side_effect = [
            _make_response(HTTPStatus.OK, json_data={"status": "SUCCESS"}),
            _make_response(HTTPStatus.OK, content=_make_result_content()),
        ]

        context = OptQCSubmissionContext(job_id="job-001")
        result = direct_client.get_job_result(context)

        assert isinstance(result, OptQCClientResult)
        assert result.machinery_result is not None
        assert _requested_url(mock_request) == "https://scheduler.local/v1/direct/jobs/job-001/result"

    @patch("claveles.client.optqc_common.requests.request")
    def test_get_result_invalid_json(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test result retrieval with invalid JSON content."""
        mock_request.side_effect = [
            _make_response(HTTPStatus.OK, json_data={"status": "SUCCESS"}),
            _make_response(HTTPStatus.OK, content=b"not json"),
        ]

        context = OptQCSubmissionContext(job_id="job-001")
        with pytest.raises(ClientServerError, match="Failed to decode"):
            direct_client.get_job_result(context)

    @patch("claveles.client.optqc_common.requests.request")
    def test_get_result_of_a_failed_job_names_the_status(
        self, mock_request: MagicMock, direct_client: OptQCClient
    ) -> None:
        """A job that failed is reported as such, the same way the cloud route reports it."""
        mock_request.return_value = _make_response(HTTPStatus.OK, json_data={"status": "FAILURE"})

        context = OptQCSubmissionContext(job_id="job-001")
        with pytest.raises(ClientServerError, match=r"not successful.*FAILURE"):
            direct_client.get_job_result(context)

    @patch("claveles.client.optqc_common.requests.request")
    def test_get_result_with_a_payload_that_is_not_a_machinery_result(
        self, mock_request: MagicMock, direct_client: OptQCClient
    ) -> None:
        """A well-formed JSON body missing the result fields is reported, not escaped as a KeyError."""
        mock_request.side_effect = [
            _make_response(HTTPStatus.OK, json_data={"status": "SUCCESS"}),
            _make_response(HTTPStatus.OK, content=b'{"unexpected": true}'),
        ]

        context = OptQCSubmissionContext(job_id="job-001")
        with pytest.raises(ClientServerError, match="Failed to decode"):
            direct_client.get_job_result(context)


# --- Cloud Backend ---


class TestCloudBackendHealth:
    """Test health endpoint for cloud backend."""

    @patch("claveles.client.optqc_common.requests.request")
    def test_health_ok(self, mock_request: MagicMock, cloud_client: OptQCClient) -> None:
        """Test health check success."""
        mock_request.return_value = _make_response(HTTPStatus.OK)
        assert cloud_client.health() is True


class TestCloudBackendSubmitJob:
    """Test job submission for cloud backend."""

    @patch("claveles.client.optqc_client.requests.put")
    @patched_machinery_repr
    @patch("claveles.client.optqc_common.requests.request")
    def test_three_step_submission(
        self, mock_request: MagicMock, mock_put: MagicMock, cloud_client: OptQCClient
    ) -> None:
        """Test three-step submission (S3 upload flow)."""
        # Step 1: upload-url returns job_id and upload_url
        upload_resp = _make_response(
            HTTPStatus.OK,
            json_data={
                "job_id": "cloud-job-001",
                "upload_url": "https://s3.example.com/upload",
            },
        )
        # Step 3: register job
        register_resp = _make_response(HTTPStatus.OK)

        mock_request.side_effect = [upload_resp, register_resp]
        mock_put.return_value = _make_response(HTTPStatus.OK)

        machinery = DummyMachineryRepr()

        with patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={"test": True}):
            context = cloud_client.submit_job(machinery)  # ty: ignore[invalid-argument-type]

        assert context.job_id == "cloud-job-001"
        mock_put.assert_called_once()

    @patch("claveles.client.optqc_client.requests.put")
    @patched_machinery_repr
    @patch("claveles.client.optqc_common.requests.request")
    def test_upload_failure(self, mock_request: MagicMock, mock_put: MagicMock, cloud_client: OptQCClient) -> None:
        """Test submission failure during upload."""
        upload_resp = _make_response(
            HTTPStatus.OK,
            json_data={
                "job_id": "cloud-job-002",
                "upload_url": "https://s3.example.com/upload",
            },
        )
        mock_request.return_value = upload_resp
        mock_put.side_effect = requests.RequestException("Upload failed")

        machinery = DummyMachineryRepr()

        with (
            patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}),
            pytest.raises(OptQCServerError, match="Failed to upload"),
        ):
            cloud_client.submit_job(machinery)  # ty: ignore[invalid-argument-type]


class TestCloudBackendGetJobResult:
    """Test result retrieval for cloud backend."""

    @patch("claveles.client.optqc_client.requests.get")
    @patch("claveles.client.optqc_common.requests.request")
    def test_successful_download(
        self, mock_request: MagicMock, mock_get: MagicMock, cloud_client: OptQCClient
    ) -> None:
        """Test successful result download."""
        content = _make_result_content()
        job_resp = _make_response(
            HTTPStatus.OK,
            json_data={
                "status": "SUCCESS",
                "download_url": "https://s3.example.com/result.gz",
            },
        )
        mock_request.return_value = job_resp
        mock_get.return_value = _make_response(HTTPStatus.OK, content=content)

        context = OptQCSubmissionContext(job_id="cloud-job-001")
        result = cloud_client.get_job_result(context)

        assert isinstance(result, OptQCClientResult)
        assert result.machinery_result is not None

    @patch("claveles.client.optqc_common.requests.request")
    def test_not_success_raises(self, mock_request: MagicMock, cloud_client: OptQCClient) -> None:
        """Test result retrieval when job is not successful."""
        job_resp = _make_response(HTTPStatus.OK, json_data={"status": "RUNNING"})
        mock_request.return_value = job_resp

        context = OptQCSubmissionContext(job_id="cloud-job-001")
        with pytest.raises(OptQCServerError, match="not successful"):
            cloud_client.get_job_result(context)

    @patch("claveles.client.optqc_common.requests.request")
    def test_missing_download_url_raises(self, mock_request: MagicMock, cloud_client: OptQCClient) -> None:
        """Test result retrieval when download URL is missing."""
        job_resp = _make_response(HTTPStatus.OK, json_data={"status": "SUCCESS"})
        mock_request.return_value = job_resp

        context = OptQCSubmissionContext(job_id="cloud-job-001")
        with pytest.raises(OptQCServerError, match="Download URL not found"):
            cloud_client.get_job_result(context)


# --- HTTP Error Handling (_request) ---


class TestRequestErrorHandling:
    """Test HTTP error classification in _request."""

    @patch("claveles.client.optqc_common.requests.request")
    def test_connection_error(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test connection failure."""
        mock_request.side_effect = requests.ConnectionError("Connection refused")
        with pytest.raises(OptQCServerError, match="Failed to connect"):
            direct_client.health()

    @patch("claveles.client.optqc_common.requests.request")
    def test_unauthorized(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test authentication error (401)."""
        mock_request.return_value = _make_response(HTTPStatus.UNAUTHORIZED)
        with pytest.raises(OptQCAuthError, match="Authentication error"):
            direct_client.health()

    @patch("claveles.client.optqc_common.requests.request")
    def test_forbidden(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test authentication error (403)."""
        mock_request.return_value = _make_response(HTTPStatus.FORBIDDEN)
        with pytest.raises(OptQCAuthError, match="Authentication error"):
            direct_client.health()

    @patch("claveles.client.optqc_common.requests.request")
    def test_bad_request(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test client error (4xx)."""
        mock_request.return_value = _make_response(HTTPStatus.BAD_REQUEST)
        with pytest.raises(OptQCClientError, match="Client error"):
            direct_client._request("GET", "/test")  # noqa: SLF001

    @patch("claveles.client.optqc_common.requests.request")
    def test_server_error(self, mock_request: MagicMock, direct_client: OptQCClient) -> None:
        """Test server error (5xx)."""
        mock_request.return_value = _make_response(HTTPStatus.INTERNAL_SERVER_ERROR)
        with pytest.raises(OptQCServerError, match="Server error"):
            direct_client._request("GET", "/test")  # noqa: SLF001


# --- run() Polling Loop ---


class TestRunPolling:
    """Test the synchronous run() method."""

    @patch("claveles.client.optqc_common.sleep")
    @patched_machinery_repr
    @patch("claveles.client.optqc_common.requests.request")
    def test_run_polls_until_success(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,
        direct_client: OptQCClient,
    ) -> None:
        """Test run() polling loop until success."""
        submit_resp = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "poll-job"})
        status_queued = _make_response(HTTPStatus.OK, json_data={"status": "QUEUED"})
        status_running = _make_response(HTTPStatus.OK, json_data={"status": "RUNNING"})
        status_success = _make_response(HTTPStatus.OK, json_data={"status": "SUCCESS"})
        result_resp = _make_response(HTTPStatus.OK, content=_make_result_content())

        mock_request.side_effect = [
            submit_resp,
            status_queued,
            status_running,
            status_success,
            # get_job_result re-reads the status before downloading, as the cloud route does.
            status_success,
            result_resp,
        ]

        machinery = DummyMachineryRepr()

        with patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}):
            result = direct_client.run(machinery)  # ty: ignore[invalid-argument-type]

        assert isinstance(result, OptQCClientResult)
        assert mock_sleep.call_count == 2  # Called for QUEUED and RUNNING

    @patch("claveles.client.optqc_common.sleep")
    @patched_machinery_repr
    @patch("claveles.client.optqc_common.requests.request")
    def test_run_raises_on_failure(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,
        direct_client: OptQCClient,
    ) -> None:
        """Test run() failure."""
        submit_resp = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "fail-job"})
        status_failure = _make_response(HTTPStatus.OK, json_data={"status": "FAILURE"})

        mock_request.side_effect = [submit_resp, status_failure]

        machinery = DummyMachineryRepr()

        with (
            patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}),
            pytest.raises(OptQCJobError, match="failed with status FAILURE"),
        ):
            direct_client.run(machinery)  # ty: ignore[invalid-argument-type]
        assert mock_sleep.call_count >= 0

    @patch("claveles.client.optqc_common.sleep")
    @patched_machinery_repr
    @patch("claveles.client.optqc_common.requests.request")
    def test_run_raises_on_timeout(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,
        direct_client: OptQCClient,
    ) -> None:
        """Test run() timeout."""
        submit_resp = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "timeout-job"})
        status_timeout = _make_response(HTTPStatus.OK, json_data={"status": "TIMEOUT"})

        mock_request.side_effect = [submit_resp, status_timeout]

        machinery = DummyMachineryRepr()

        with (
            patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}),
            pytest.raises(OptQCJobError, match="failed with status TIMEOUT"),
        ):
            direct_client.run(machinery)  # ty: ignore[invalid-argument-type]
        assert mock_sleep.call_count >= 0


class TestWaitForCompletion:
    """Test wait_for_completion polling."""

    @patch("claveles.client.optqc_common.sleep")
    @patch("claveles.client.optqc_common.requests.request")
    def test_wait_for_completion_success(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,
        direct_client: OptQCClient,
    ) -> None:
        """Test polling until SUCCESS."""
        status_queued = _make_response(HTTPStatus.OK, json_data={"status": "QUEUED"})
        status_running = _make_response(HTTPStatus.OK, json_data={"status": "RUNNING"})
        status_success = _make_response(HTTPStatus.OK, json_data={"status": "SUCCESS"})
        mock_request.side_effect = [status_queued, status_running, status_success]

        context = OptQCSubmissionContext(job_id="poll-job")
        status = direct_client.wait_for_completion(context)

        assert status == "SUCCESS"
        assert mock_sleep.call_count == 2

    @patch("claveles.client.optqc_common.sleep")
    @patch("claveles.client.optqc_common.requests.request")
    def test_wait_for_completion_cancelled(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,
        direct_client: OptQCClient,
    ) -> None:
        """Test polling until CANCELLED."""
        status_running = _make_response(HTTPStatus.OK, json_data={"status": "RUNNING"})
        status_cancelled = _make_response(HTTPStatus.OK, json_data={"status": "CANCELLED"})
        mock_request.side_effect = [status_running, status_cancelled]

        context = OptQCSubmissionContext(job_id="cancel-job")
        status = direct_client.wait_for_completion(context)

        assert status == "CANCELLED"
        assert mock_sleep.call_count == 1

    @patch("claveles.client.optqc_common.sleep")
    @patch("claveles.client.optqc_common.monotonic")
    @patch("claveles.client.optqc_common.requests.request")
    def test_wait_for_completion_exceeds_polling_timeout(
        self,
        mock_request: MagicMock,
        mock_monotonic: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Test that JobTimeoutError is raised when polling timeout is exceeded."""
        client = OptQCClient(
            url="https://scheduler.local",
            api_key="test-api-key",
            target="direct",
            qc_node_id="node-001",
            polling_timeout_msec=1000,
            polling_interval=0.01,
        )
        mock_request.return_value = _make_response(HTTPStatus.OK, json_data={"status": "QUEUED"})
        # First call: within deadline; second call: past deadline
        mock_monotonic.side_effect = [0.0, 0.5, 2.0]

        context = OptQCSubmissionContext(job_id="hang-job")
        with pytest.raises(JobTimeoutError, match="polling timeout") as error:
            client.wait_for_completion(context)
        # An expired wait reads as a TimeoutError, whichever client did the waiting.
        assert isinstance(error.value, TimeoutError)
        assert mock_sleep.call_count >= 0

    @patch("claveles.client.optqc_common.sleep")
    @patch("claveles.client.optqc_common.requests.request")
    def test_wait_for_completion_lowercase_status(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,
        direct_client: OptQCClient,
    ) -> None:
        """Test that lowercase status strings are treated as terminal states."""
        mock_request.return_value = _make_response(HTTPStatus.OK, json_data={"status": "success"})

        context = OptQCSubmissionContext(job_id="lower-job")
        status = direct_client.wait_for_completion(context)

        assert status == "SUCCESS"
        assert mock_sleep.call_count == 0

    @patch("claveles.client.optqc_common.sleep")
    @patch("claveles.client.optqc_common.requests.request")
    def test_wait_for_completion_logs_status_changes(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,  # noqa: ARG002
        direct_client: OptQCClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that status changes are logged at INFO level."""
        mock_request.side_effect = [
            _make_response(HTTPStatus.OK, json_data={"status": "QUEUED"}),
            _make_response(HTTPStatus.OK, json_data={"status": "RUNNING"}),
            _make_response(HTTPStatus.OK, json_data={"status": "SUCCESS"}),
        ]

        context = OptQCSubmissionContext(job_id="log-job")
        with caplog.at_level(logging.INFO, logger="claveles.client.optqc_common"):
            direct_client.wait_for_completion(context)

        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("Waiting for job" in m for m in messages)
        assert any("QUEUED" in m for m in messages)
        assert any("RUNNING" in m for m in messages)
        assert any("SUCCESS" in m for m in messages)

    @patch("claveles.client.optqc_common.sleep")
    @patch("claveles.client.optqc_common.requests.request")
    def test_wait_for_completion_no_duplicate_log_for_same_status(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,  # noqa: ARG002
        direct_client: OptQCClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that repeated identical statuses produce only one status log entry."""
        mock_request.side_effect = [
            _make_response(HTTPStatus.OK, json_data={"status": "QUEUED"}),
            _make_response(HTTPStatus.OK, json_data={"status": "QUEUED"}),
            _make_response(HTTPStatus.OK, json_data={"status": "SUCCESS"}),
        ]

        context = OptQCSubmissionContext(job_id="dedup-job")
        with caplog.at_level(logging.INFO, logger="claveles.client.optqc_common"):
            direct_client.wait_for_completion(context)

        status_logs = [r.getMessage() for r in caplog.records if "status:" in r.getMessage().lower()]
        queued_logs = [m for m in status_logs if "QUEUED" in m]
        assert len(queued_logs) == 1


class TestRunCaseInsensitive:
    """Test that run() accepts lowercase terminal status strings."""

    @patch("claveles.client.optqc_common.sleep")
    @patched_machinery_repr
    @patch("claveles.client.optqc_common.requests.request")
    def test_run_succeeds_with_lowercase_success(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,  # noqa: ARG002
        direct_client: OptQCClient,
    ) -> None:
        """Test that run() treats 'success' (lowercase) as a successful terminal status."""
        submit_resp = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "lower-job"})
        status_success = _make_response(HTTPStatus.OK, json_data={"status": "success"})
        result_resp = _make_response(HTTPStatus.OK, content=_make_result_content())
        # The second status response is the one get_job_result reads before downloading.
        mock_request.side_effect = [submit_resp, status_success, status_success, result_resp]

        machinery = DummyMachineryRepr()
        with patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}):
            result = direct_client.run(machinery)  # ty: ignore[invalid-argument-type]

        assert isinstance(result, OptQCClientResult)


class TestCloudBackendGetJobResultCaseInsensitive:
    """Test that cloud get_job_result() accepts lowercase status strings."""

    @patch("claveles.client.optqc_client.requests.get")
    @patch("claveles.client.optqc_common.requests.request")
    def test_get_result_lowercase_success(
        self,
        mock_request: MagicMock,
        mock_get: MagicMock,
        cloud_client: OptQCClient,
    ) -> None:
        """Test that 'success' (lowercase) is accepted as a successful status in get_job_result."""
        content = _make_result_content()
        job_resp = _make_response(
            HTTPStatus.OK,
            json_data={
                "status": "success",
                "download_url": "https://s3.example.com/result.gz",
            },
        )
        mock_request.return_value = job_resp
        mock_get.return_value = _make_response(HTTPStatus.OK, content=content)

        context = OptQCSubmissionContext(job_id="cloud-lower-job")
        result = cloud_client.get_job_result(context)

        assert isinstance(result, OptQCClientResult)
        assert result.machinery_result is not None
