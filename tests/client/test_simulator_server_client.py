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

"""Tests for the Cloud Simulator Server (CSS) direct client (mock HTTP).

These tests mock ``requests.request`` and never import ``claveles_simulator``, so they run without the
simulator wheel installed -- verifying that ``mode="simulator-server"`` genuinely works standalone.
"""

from __future__ import annotations

from datetime import timedelta
from http import HTTPStatus
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic
from claveles.circuit.state import BosonicState
from claveles.client import (
    OptQCAuthError,
    OptQCServerError,
    SimulatorClient,
    SimulatorServerBusyError,
    SimulatorServerConflictError,
    SimulatorServerGoneError,
)
from claveles.graph import GraphRepr
from claveles.machinery import MachineryRepr
from claveles.pb.claveles_cloud.program.v1.graph_pb2 import GraphResult as PbGraphResult
from claveles.pb.claveles_cloud.program.v1.machinery_pb2 import MachineryResult as PbMachineryResult
from claveles.pb.claveles_cloud.program.v1.quantum_program_pb2 import QuantumProgramResult


def _make_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = "" if json_data is None else str(json_data)
    resp.headers = headers or {}
    return resp


def _make_circuit() -> CircuitRepr:
    """A trivial single-mode bosonic circuit, fully initialized."""
    circuit = CircuitRepr("css_test_circuit")
    circuit.Q(0) | BosonicState.vacuum()
    circuit.Q(0) | intrinsic.Displacement(1.0, 0.0)
    return circuit


def _success_detail(resolved_backend: str, effective_seed: int = 0) -> dict[str, Any]:
    """A terminal ``JobDetailResponse``-shaped dict with status SUCCESS."""
    return {
        "status": "SUCCESS",
        "resolved_backend": resolved_backend,
        "effective_seed": effective_seed,
        "timeline": {},
    }


def _envelope(
    payload_proto: QuantumProgramResult | PbGraphResult | PbMachineryResult | None = None,
    program_type: str = "CIRCUIT",
) -> dict[str, Any]:
    """A ``ResultEnvelope``-shaped dict as ``requests`` would hand back after gzip decoding."""
    import base64  # noqa: PLC0415

    proto = payload_proto if payload_proto is not None else QuantumProgramResult()
    return {
        "program_type": program_type,
        "simulator_version": "0.1.0-test",
        "payload": base64.b64encode(proto.SerializeToString()).decode("ascii"),
    }


@pytest.fixture
def client() -> SimulatorClient:
    return SimulatorClient(
        n_shots=1,
        mode="simulator-server",
        url="https://css.local",
        api_key="css-test-token",
        polling_interval=0.0,
    )


class TestConstructorValidation:
    """Mode/parameter validation that does not need any HTTP mocking."""

    def test_simulator_server_requires_url_and_api_key(self) -> None:
        with pytest.raises(ValueError, match="requires url and api_key"):
            SimulatorClient(mode="simulator-server")

    def test_simulator_server_rejects_state_save_policy(self) -> None:
        with pytest.raises(ValueError, match="state_save_policy"):
            SimulatorClient(mode="simulator-server", url="u", api_key="k", state_save_policy="all")

    def test_local_rejects_url_and_api_key(self) -> None:
        with pytest.raises(ValueError, match="mode='local'"):
            SimulatorClient(mode="local", url="u", api_key="k")

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            SimulatorClient(mode="bogus")  # ty: ignore[invalid-argument-type]

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPTQC_SIMULATOR_URL", "https://from-env.local")
        monkeypatch.setenv("OPTQC_SIMULATOR_API_KEY", "env-token")
        sdk_client = SimulatorClient(mode="simulator-server")
        assert sdk_client.url == "https://from-env.local"
        assert sdk_client.api_key == "env-token"


class TestRequestBody:
    """The JSON body POSTed to /v1/jobs."""

    @patch("claveles.client._simulator_server.requests.request")
    def test_backend_omitted_when_unspecified(self, mock_request: MagicMock, client: SimulatorClient) -> None:
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, _success_detail("single_gpu")),
            _make_response(HTTPStatus.OK, _envelope()),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        client.run(_make_circuit())

        post_kwargs = mock_request.call_args_list[0].kwargs
        body = post_kwargs["json"]
        assert "backend" not in body
        assert body["program_type"] == "CIRCUIT"
        assert body["n_shots"] == 1
        assert "job_id" in body
        assert "lease_id" in body
        assert "state_save_policy" not in body

    @patch("claveles.client._simulator_server.requests.request")
    def test_backend_included_when_explicit(self, mock_request: MagicMock) -> None:
        explicit_client = SimulatorClient(
            n_shots=1, backend="single_cpu", mode="simulator-server", url="https://css.local", api_key="k"
        )
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, _success_detail("single_cpu")),
            _make_response(HTTPStatus.OK, _envelope()),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        explicit_client.run(_make_circuit())

        assert mock_request.call_args_list[0].kwargs["json"]["backend"] == "single_cpu"

    @patch("claveles.client._simulator_server.requests.request")
    def test_backend_cpu_alias_mapped_to_multi_cpu(self, mock_request: MagicMock) -> None:
        """CSS's Backend enum has no "cpu" entry; the client must map it like local mode does."""
        cpu_client = SimulatorClient(
            n_shots=1, backend="cpu", mode="simulator-server", url="https://css.local", api_key="k"
        )
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, _success_detail("multi_cpu")),
            _make_response(HTTPStatus.OK, _envelope()),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        cpu_client.run(_make_circuit())

        assert mock_request.call_args_list[0].kwargs["json"]["backend"] == "multi_cpu"


class TestPollingAndResult:
    """The submit -> poll -> fetch-result flow."""

    @patch("claveles.client._simulator_server.sleep")
    @patch("claveles.client._simulator_server.requests.request")
    def test_polls_until_success_and_decodes_result(
        self, mock_request: MagicMock, mock_sleep: MagicMock, client: SimulatorClient
    ) -> None:
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, {"status": "QUEUED", "queue_position": 1}),
            _make_response(HTTPStatus.OK, {"status": "RUNNING"}),
            _make_response(
                HTTPStatus.OK,
                {
                    "status": "SUCCESS",
                    "resolved_backend": "single_gpu",
                    "effective_seed": 42,
                    "timeline": {
                        "queued_at": "2026-01-01T00:00:00Z",
                        "started_at": "2026-01-01T00:00:01Z",
                        "finished_at": "2026-01-01T00:00:02Z",
                    },
                },
            ),
            _make_response(HTTPStatus.OK, _envelope()),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        result = client.run(_make_circuit())

        assert result.circuit_result is not None
        assert result.execution_details.simulator_version == "0.1.0-test"
        assert mock_sleep.call_count == 2  # QUEUED, RUNNING

    @patch("claveles.client._simulator_server.sleep")
    @patch("claveles.client._simulator_server.requests.request")
    def test_timeline_shares_one_timezone(self, mock_request: MagicMock, mock_sleep: MagicMock) -> None:  # noqa: ARG002
        """CSS's timestamps parse as UTC-aware; they must be converted to the client's own tz too."""
        from zoneinfo import ZoneInfo  # noqa: PLC0415

        tokyo_client = SimulatorClient(
            n_shots=1,
            mode="simulator-server",
            url="https://css.local",
            api_key="k",
            timezone=ZoneInfo("Asia/Tokyo"),
        )
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(
                HTTPStatus.OK,
                {
                    "status": "SUCCESS",
                    "resolved_backend": "single_gpu",
                    "effective_seed": 42,
                    "timeline": {
                        "queued_at": "2026-01-01T00:00:00Z",
                        "started_at": "2026-01-01T00:00:01Z",
                        "finished_at": "2026-01-01T00:00:02Z",
                    },
                },
            ),
            _make_response(HTTPStatus.OK, _envelope()),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        result = tokyo_client.run(_make_circuit())

        timeline = result.execution_details.timeline
        assert timeline is not None
        tzinfos = {
            dt.tzinfo
            for dt in (
                timeline.submitted_at,
                timeline.queued_at,
                timeline.dequeued_at,
                timeline.execution_started_at,
                timeline.execution_finished_at,
                timeline.finished_at,
            )
            if dt is not None
        }
        assert len(tzinfos) == 1
        assert timeline.queued_at is not None
        assert timeline.queued_at.utcoffset() == timedelta(hours=9)

    @patch("claveles.client._simulator_server.sleep")
    @patch("claveles.client._simulator_server.requests.request")
    def test_failure_raises_runtime_error_with_mapped_id(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,  # noqa: ARG002
        client: SimulatorClient,
    ) -> None:
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(
                HTTPStatus.OK,
                {
                    "status": "FAILURE",
                    "error": {"code": "LARGE_INDEX", "message": "boom"},
                },
            ),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        with pytest.raises(RuntimeError) as exc_info:
            client.run(_make_circuit())

        assert exc_info.value.args == ("boom", 301)

    @patch("claveles.client._simulator_server.sleep")
    @patch("claveles.client._simulator_server.requests.request")
    def test_timeout_status_maps_to_207(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,  # noqa: ARG002
        client: SimulatorClient,
    ) -> None:
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, {"status": "TIMEOUT", "error": {"code": "TIMEOUT", "message": "killed"}}),
            _make_response(HTTPStatus.NO_CONTENT),
        ]

        with pytest.raises(RuntimeError) as exc_info:
            client.run(_make_circuit())

        assert exc_info.value.args == ("killed", 207)

    @patch("claveles.client._simulator_server.monotonic")
    @patch("claveles.client._simulator_server.sleep")
    @patch("claveles.client._simulator_server.requests.request")
    def test_polling_deadline_exceeded_raises_timeout_error(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,  # noqa: ARG002
        mock_monotonic: MagicMock,
        client: SimulatorClient,
    ) -> None:
        # First call establishes the deadline; the second (inside the loop check) is already past it.
        mock_monotonic.side_effect = [0.0, 10_000_000.0]
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
        ]

        with pytest.raises(TimeoutError):
            client.run(_make_circuit())


class TestErrorMapping:
    """HTTP status code -> exception mapping for the transport layer."""

    @patch("claveles.client._simulator_server.requests.request")
    def test_401_raises_auth_error(self, mock_request: MagicMock, client: SimulatorClient) -> None:
        mock_request.return_value = _make_response(HTTPStatus.UNAUTHORIZED, {"code": "UNAUTHORIZED", "message": "no"})
        with pytest.raises(OptQCAuthError):
            client.run(_make_circuit())

    @patch("claveles.client._simulator_server.requests.request")
    def test_409_raises_conflict_error(self, mock_request: MagicMock, client: SimulatorClient) -> None:
        mock_request.return_value = _make_response(
            HTTPStatus.CONFLICT, {"code": "JOB_CONTENT_CONFLICT", "message": "no"}
        )
        with pytest.raises(SimulatorServerConflictError):
            client.run(_make_circuit())

    @patch("claveles.client._simulator_server.requests.request")
    def test_410_raises_gone_error(self, mock_request: MagicMock, client: SimulatorClient) -> None:
        mock_request.return_value = _make_response(HTTPStatus.GONE, {"code": "GONE", "message": "no"})
        with pytest.raises(SimulatorServerGoneError):
            client.run(_make_circuit())

    @patch("claveles.client._simulator_server.requests.request")
    def test_429_raises_busy_error_with_retry_after(self, mock_request: MagicMock, client: SimulatorClient) -> None:
        mock_request.return_value = _make_response(
            HTTPStatus.TOO_MANY_REQUESTS, {"code": "QUEUE_FULL", "message": "no"}, headers={"Retry-After": "2"}
        )
        with pytest.raises(SimulatorServerBusyError) as exc_info:
            client.run(_make_circuit())
        assert exc_info.value.retry_after == pytest.approx(2.0)

    @patch("claveles.client._simulator_server.requests.request")
    def test_500_raises_server_error(self, mock_request: MagicMock, client: SimulatorClient) -> None:
        mock_request.return_value = _make_response(
            HTTPStatus.INTERNAL_SERVER_ERROR, {"code": "INTERNAL", "message": "no"}
        )
        with pytest.raises(OptQCServerError):
            client.run(_make_circuit())

    @patch("claveles.client._simulator_server.sleep")
    @patch("claveles.client._simulator_server.requests.request")
    def test_connection_error_raises_server_error(
        self, mock_request: MagicMock, mock_sleep: MagicMock, client: SimulatorClient
    ) -> None:
        mock_request.side_effect = requests.ConnectionError("refused")
        with pytest.raises(OptQCServerError):
            client.run(_make_circuit())
        assert mock_sleep.call_count == 2  # exhausts all 3 submit attempts

    @patch("claveles.client._simulator_server.sleep")
    @patch("claveles.client._simulator_server.requests.request")
    def test_submit_retries_after_transient_connection_error(
        self, mock_request: MagicMock, mock_sleep: MagicMock, client: SimulatorClient
    ) -> None:
        """A lost response to the submit POST must not orphan the job -- retry under the same job_id."""
        mock_request.side_effect = [
            requests.ConnectionError("refused"),
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, _success_detail("single_gpu")),
            _make_response(HTTPStatus.OK, _envelope()),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        client.run(_make_circuit())

        assert mock_sleep.call_count == 1
        first_attempt_job_id = mock_request.call_args_list[0].kwargs["json"]["job_id"]
        second_attempt_job_id = mock_request.call_args_list[1].kwargs["json"]["job_id"]
        assert first_attempt_job_id == second_attempt_job_id


class TestGraphAndMachineryRepresentations:
    """CSS supports GRAPH/MACHINERY too; squeezing_level is required, circuit-only params are not."""

    @patch("claveles.client._simulator_server.requests.request")
    def test_graph_repr_body_and_decode(self, mock_request: MagicMock) -> None:
        sdk_client = SimulatorClient(
            n_shots=1,
            resource_squeezing_level=8.0,
            mode="simulator-server",
            url="https://css.local",
            api_key="k",
        )
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, _success_detail("single_gpu")),
            _make_response(HTTPStatus.OK, _envelope(PbGraphResult(), program_type="GRAPH")),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        result = sdk_client.run(GraphRepr(1, 1))

        body = mock_request.call_args_list[0].kwargs["json"]
        assert body["program_type"] == "GRAPH"
        assert body["squeezing_level"] == pytest.approx(8.0)
        assert "svd_cutoff" not in body
        assert "track_truncation_error" not in body
        assert "max_bond_dim" not in body
        assert result.graph_result is not None

    @patch("claveles.client._simulator_server.requests.request")
    def test_machinery_repr_body_and_decode(self, mock_request: MagicMock) -> None:
        sdk_client = SimulatorClient(
            n_shots=1,
            resource_squeezing_level=8.0,
            mode="simulator-server",
            url="https://css.local",
            api_key="k",
        )
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, _success_detail("single_gpu")),
            _make_response(HTTPStatus.OK, _envelope(PbMachineryResult(), program_type="MACHINERY")),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        result = sdk_client.run(MachineryRepr(1, 1))

        body = mock_request.call_args_list[0].kwargs["json"]
        assert body["program_type"] == "MACHINERY"
        assert body["squeezing_level"] == pytest.approx(8.0)
        assert "svd_cutoff" not in body
        assert "track_truncation_error" not in body
        assert "max_bond_dim" not in body
        assert result.machinery_result is not None

    @patch("claveles.client._simulator_server.requests.request")
    def test_circuit_repr_body_omits_squeezing_level(self, mock_request: MagicMock, client: SimulatorClient) -> None:
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, _success_detail("single_gpu")),
            _make_response(HTTPStatus.OK, _envelope()),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        client.run(_make_circuit())

        body = mock_request.call_args_list[0].kwargs["json"]
        assert "squeezing_level" not in body
        assert body["svd_cutoff"] == pytest.approx(0.0)
        assert body["track_truncation_error"] is False
