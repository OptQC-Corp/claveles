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

"""Tests for the Cloud User API remote client (mock HTTP), ``SimulatorClient(mode="remote", ...)``.

These mock ``requests.request``/``requests.put``/``requests.get`` directly and never import
``claveles_simulator``, so they run without the simulator wheel installed. Unlike
``mode="simulator-server"`` (:mod:`claveles.client._simulator_server`, tested in
``test_simulator_server_client.py``), this path is a three-step S3 flow (Step 1 ``POST`` through
``_request``/``requests.request``, Step 2 ``PUT`` the raw program via ``requests.put``, Step 3
``POST`` the settings through ``_request``), and its result is downloaded from a presigned URL via
``requests.get`` rather than through the polling endpoint's own transport.
"""

from __future__ import annotations

import base64
import gzip
import json
from http import HTTPStatus
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import BosonicState
from claveles.client import JobFailedError, SimulatorClient
from claveles.graph import GraphRepr
from claveles.machinery import MachineryRepr
from claveles.pb.claveles_cloud.program.v1.graph_pb2 import GraphResult as PbGraphResult
from claveles.pb.claveles_cloud.program.v1.machinery_pb2 import MachineryResult as PbMachineryResult
from claveles.pb.claveles_cloud.program.v1.quantum_program_pb2 import QuantumProgramResult
from claveles.warnings import IgnoredOptionWarning


def _circuit() -> CircuitRepr:
    circuit = CircuitRepr("remote")
    circuit.Q(0) | BosonicState.vacuum()
    circuit.Q(0) | std.MeasureHomodyne(0.0)
    return circuit


def _make_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    content: bytes = b"",
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.content = content
    resp.text = "" if json_data is None else str(json_data)
    resp.headers = {}
    resp.raise_for_status = MagicMock()
    return resp


def _envelope_bytes(program_type: str, *, gzipped: bool = True) -> bytes:
    """Serialize a ``ResultEnvelope``-shaped JSON body, matching what CSS itself produces.

    Args:
        program_type: ``"CIRCUIT"``/``"GRAPH"``/``"MACHINERY"``.
        gzipped: Whether to gzip-compress the body, matching the S3 object
            ``simulator-bridge``/``bridge`` relay it as. ``False`` exercises the branch where a plain
            HTTP client already decompressed it on the way through (see the module docstring of
            ``claveles.client._simulator_remote``).
    """
    payload = {
        "CIRCUIT": QuantumProgramResult,
        "GRAPH": PbGraphResult,
        "MACHINERY": PbMachineryResult,
    }[program_type]()
    envelope = {
        "program_type": program_type,
        "simulator_version": "0.1.0-test",
        "payload": base64.b64encode(payload.SerializeToString()).decode("ascii"),
    }
    body = json.dumps(envelope).encode("utf-8")
    return gzip.compress(body) if gzipped else body


def _create_response(job_id: str = "j1") -> MagicMock:
    return _make_response(
        HTTPStatus.OK,
        {"job_id": job_id, "upload_url": "https://s3.local/upload", "expires_at": "2026-01-01T00:00:00Z"},
    )


def _success_detail(*, download_url: str = "https://s3.local/download") -> dict[str, Any]:
    return {
        "job_id": "j1",
        "status": "SUCCESS",
        "execution_report": {"queue_time_msec": 10, "execution_time_msec": 20, "simulator_version": "0.1.0-test"},
        "download_url": download_url,
    }


def _remote_client(**kwargs: Any) -> SimulatorClient:  # noqa: ANN401  # forwards whichever option is under test
    return SimulatorClient(
        n_shots=1,
        mode="remote",
        url="https://compute.cloud.optqc.com",
        api_key="k",
        polling_interval=0.0,
        **kwargs,
    )


class TestConstructionIsValidatedTheSameWayAsSimulatorServer:
    def test_missing_url_and_api_key_are_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPTQC_CLIENT_URL", raising=False)
        monkeypatch.delenv("OPTQC_CLIENT_API_KEY", raising=False)

        with pytest.raises(ValueError, match="requires url and api_key"):
            SimulatorClient(mode="remote")

    def test_it_does_not_fall_back_to_the_simulator_server_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPTQC_CLIENT_URL", raising=False)
        monkeypatch.delenv("OPTQC_CLIENT_API_KEY", raising=False)
        monkeypatch.setenv("OPTQC_SIMULATOR_URL", "https://css.local")
        monkeypatch.setenv("OPTQC_SIMULATOR_API_KEY", "css-key")

        with pytest.raises(ValueError, match="requires url and api_key"):
            SimulatorClient(mode="remote")

    def test_the_optqc_client_env_vars_are_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPTQC_CLIENT_URL", "https://from-env.local")
        monkeypatch.setenv("OPTQC_CLIENT_API_KEY", "env-key")

        client = SimulatorClient(mode="remote")

        assert client.url == "https://from-env.local"
        assert client.api_key == "env-key"

    def test_state_save_policy_cannot_be_set(self) -> None:
        with pytest.raises(ValueError, match="state_save_policy cannot be set for mode='remote'"):
            _remote_client(state_save_policy="all")

    def test_max_saved_state_elements_cannot_be_set(self) -> None:
        with pytest.raises(ValueError, match="max_saved_state_elements cannot be set for mode='remote'"):
            _remote_client(max_saved_state_elements=4096)

    def test_an_unspecified_state_save_policy_reads_as_first_only(self) -> None:
        assert _remote_client().state_save_policy == "first_only"

    def test_state_representation_other_than_auto_is_refused(self) -> None:
        with pytest.raises(ValueError, match="state_representation cannot be set to 'fock' for mode='remote'"):
            _remote_client(state_representation="fock")

    def test_auto_is_accepted(self) -> None:
        # Should not raise merely for being constructed.
        client = _remote_client(state_representation="auto")
        assert client.state_representation == "auto"


class TestStateRepresentationAssignedAfterConstructionIsAlsoRefused:
    def test_it_is_refused_before_anything_is_uploaded_for_a_circuit(self) -> None:
        # No HTTP mocking: reaching the network would mean the check came too late.
        client = _remote_client()
        client.state_representation = "fock"

        with pytest.raises(ValueError, match="state_representation is 'fock'"):
            client.run(_circuit())

    @patch("claveles.client._simulator_remote.requests.get")
    @patch("claveles.client._simulator_remote.requests.put")
    @patch("claveles.client._simulator_remote.requests.request")
    def test_it_is_not_checked_for_a_graph_run(
        self, mock_request: MagicMock, mock_put: MagicMock, mock_get: MagicMock
    ) -> None:
        # state_representation is a circuit-only concept; a graph run must not be refused for it, only
        # warned about (IgnoredOptionWarning, covered by test_ignored_options.py) since it was set but
        # unread.
        mock_request.side_effect = [
            _create_response(),
            _make_response(HTTPStatus.OK),
            _make_response(HTTPStatus.OK, _success_detail()),
        ]
        mock_put.return_value = _make_response(HTTPStatus.OK)
        mock_get.return_value = _make_response(HTTPStatus.OK, content=_envelope_bytes("GRAPH"))
        client = _remote_client()
        client.state_representation = "fock"

        with pytest.warns(IgnoredOptionWarning, match="state_representation"):
            result = client.run(GraphRepr(1, 1))

        assert result.graph_result is not None


class TestASuccessfulRoundTrip:
    @patch("claveles.client._simulator_remote.requests.get")
    @patch("claveles.client._simulator_remote.requests.put")
    @patch("claveles.client._simulator_remote.requests.request")
    def test_a_graph_run_round_trips(self, mock_request: MagicMock, mock_put: MagicMock, mock_get: MagicMock) -> None:
        mock_request.side_effect = [
            _create_response(),
            _make_response(HTTPStatus.OK),  # Step 3 submit
            _make_response(HTTPStatus.OK, _success_detail()),  # poll
        ]
        mock_put.return_value = _make_response(HTTPStatus.OK)
        mock_get.return_value = _make_response(HTTPStatus.OK, content=_envelope_bytes("GRAPH"))
        client = _remote_client(resource_squeezing_level=8.0)

        result = client.run(GraphRepr(1, 1))

        assert result.graph_result is not None
        assert result.execution_details.simulator_version == "0.1.0-test"

    @patch("claveles.client._simulator_remote.requests.get")
    @patch("claveles.client._simulator_remote.requests.put")
    @patch("claveles.client._simulator_remote.requests.request")
    def test_a_machinery_run_round_trips(
        self, mock_request: MagicMock, mock_put: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_request.side_effect = [
            _create_response(),
            _make_response(HTTPStatus.OK),
            _make_response(HTTPStatus.OK, _success_detail()),
        ]
        mock_put.return_value = _make_response(HTTPStatus.OK)
        mock_get.return_value = _make_response(HTTPStatus.OK, content=_envelope_bytes("MACHINERY"))
        client = _remote_client(resource_squeezing_level=8.0)

        result = client.run(MachineryRepr(1, 1))

        assert result.machinery_result is not None

    @patch("claveles.client._simulator_remote.requests.get")
    @patch("claveles.client._simulator_remote.requests.put")
    @patch("claveles.client._simulator_remote.requests.request")
    def test_a_circuit_run_round_trips(
        self, mock_request: MagicMock, mock_put: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_request.side_effect = [
            _create_response(),
            _make_response(HTTPStatus.OK),
            _make_response(HTTPStatus.OK, _success_detail()),
        ]
        mock_put.return_value = _make_response(HTTPStatus.OK)
        mock_get.return_value = _make_response(HTTPStatus.OK, content=_envelope_bytes("CIRCUIT"))
        client = _remote_client()

        result = client.run(_circuit())

        assert result.circuit_result is not None

    @patch("claveles.client._simulator_remote.requests.get")
    @patch("claveles.client._simulator_remote.requests.put")
    @patch("claveles.client._simulator_remote.requests.request")
    def test_the_upload_is_raw_bytes_not_base64(
        self, mock_request: MagicMock, mock_put: MagicMock, mock_get: MagicMock
    ) -> None:
        # Unlike mode="simulator-server", which base64-wraps the program inside a JSON envelope, the
        # S3 object here must be the raw serialized proto: the wire contract simulator-bridge and CSS
        # read is `simulator_jobs/{job_id}/program.pb`, not base64 text.
        mock_request.side_effect = [
            _create_response(),
            _make_response(HTTPStatus.OK),
            _make_response(HTTPStatus.OK, _success_detail()),
        ]
        mock_put.return_value = _make_response(HTTPStatus.OK)
        mock_get.return_value = _make_response(HTTPStatus.OK, content=_envelope_bytes("GRAPH"))
        client = _remote_client(resource_squeezing_level=8.0)

        graph = GraphRepr(1, 1)
        client.run(graph)

        uploaded = mock_put.call_args.kwargs["data"]
        assert uploaded == graph.proto().SerializeToString()

    @patch("claveles.client._simulator_remote.requests.get")
    @patch("claveles.client._simulator_remote.requests.put")
    @patch("claveles.client._simulator_remote.requests.request")
    def test_the_settings_field_is_named_resource_squeezing_level(
        self, mock_request: MagicMock, mock_put: MagicMock, mock_get: MagicMock
    ) -> None:
        # Unlike CSS's own JobCreateRequest ("squeezing_level"), the Cloud User API's
        # SimulatorJobExecutionSettings names this field "resource_squeezing_level".
        mock_request.side_effect = [
            _create_response(),
            _make_response(HTTPStatus.OK),
            _make_response(HTTPStatus.OK, _success_detail()),
        ]
        mock_put.return_value = _make_response(HTTPStatus.OK)
        mock_get.return_value = _make_response(HTTPStatus.OK, content=_envelope_bytes("GRAPH"))
        client = _remote_client(resource_squeezing_level=8.0)

        client.run(GraphRepr(1, 1))

        submit_call = mock_request.call_args_list[1]
        settings = submit_call.kwargs["json"]["settings"]
        assert settings["resource_squeezing_level"] == pytest.approx(8.0)
        assert "squeezing_level" not in settings

    @patch("claveles.client._simulator_remote.requests.get")
    @patch("claveles.client._simulator_remote.requests.put")
    @patch("claveles.client._simulator_remote.requests.request")
    def test_an_already_decompressed_download_is_also_accepted(
        self, mock_request: MagicMock, mock_put: MagicMock, mock_get: MagicMock
    ) -> None:
        # A presigned GET may come back pre-decompressed by an HTTP client honouring
        # Content-Encoding: gzip; the envelope must decode either way.
        mock_request.side_effect = [
            _create_response(),
            _make_response(HTTPStatus.OK),
            _make_response(HTTPStatus.OK, _success_detail()),
        ]
        mock_put.return_value = _make_response(HTTPStatus.OK)
        mock_get.return_value = _make_response(HTTPStatus.OK, content=_envelope_bytes("GRAPH", gzipped=False))
        client = _remote_client(resource_squeezing_level=8.0)

        result = client.run(GraphRepr(1, 1))

        assert result.graph_result is not None


class TestAFailedJobRaisesJobFailedError:
    @patch("claveles.client._simulator_remote.requests.put")
    @patch("claveles.client._simulator_remote.requests.request")
    def test_the_error_message_is_read_from_the_execution_report(
        self, mock_request: MagicMock, mock_put: MagicMock
    ) -> None:
        mock_request.side_effect = [
            _create_response(),
            _make_response(HTTPStatus.OK),
            _make_response(
                HTTPStatus.OK,
                {
                    "job_id": "j1",
                    "status": "FAILURE",
                    "execution_report": {
                        "queue_time_msec": 1,
                        "execution_time_msec": 1,
                        "error_message": "boom",
                    },
                },
            ),
        ]
        mock_put.return_value = _make_response(HTTPStatus.OK)
        client = _remote_client(resource_squeezing_level=8.0)

        with pytest.raises(JobFailedError, match="boom"):
            client.run(GraphRepr(1, 1))

    @patch("claveles.client._simulator_remote.requests.put")
    @patch("claveles.client._simulator_remote.requests.request")
    def test_a_missing_error_message_falls_back_to_a_generic_one(
        self, mock_request: MagicMock, mock_put: MagicMock
    ) -> None:
        mock_request.side_effect = [
            _create_response(),
            _make_response(HTTPStatus.OK),
            _make_response(HTTPStatus.OK, {"job_id": "j1", "status": "TIMEOUT"}),
        ]
        mock_put.return_value = _make_response(HTTPStatus.OK)
        client = _remote_client(resource_squeezing_level=8.0)

        with pytest.raises(JobFailedError, match="status TIMEOUT"):
            client.run(GraphRepr(1, 1))
