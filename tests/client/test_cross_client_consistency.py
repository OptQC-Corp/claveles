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

"""Tests that the same situation reads the same way whichever client and mode produced it.

Each test here pins one agreement, so that none of these becomes a disagreement: the same run raising in
one mode and returning a value in the other, the same misuse raising two different exception types, the
same omission caught at construction on one client and much later on the other, one state arriving under
two spellings.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, get_args
from unittest.mock import MagicMock, patch

import pytest
import requests

from claveles.circuit import CircuitRepr
from claveles.client import (
    ClientServerError,
    JobFailedError,
    JobStatus,
    OptQCClient,
    OptQCSubmissionContext,
    OptQCTarget,
    SimulatorClient,
    optqc_client,
)
from claveles.client.abstract import unsupported_representation_message
from claveles.graph import GraphRepr
from claveles.pb.claveles_cloud.program.v1.graph_pb2 import GraphResult as PbGraphResult


class NotARepresentation:
    """Stand-in for whatever a caller passes by mistake."""


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
    return resp


def _graph_envelope() -> dict[str, Any]:
    """A ``ResultEnvelope``-shaped dict carrying an empty graph result."""
    import base64  # noqa: PLC0415

    return {
        "program_type": "GRAPH",
        "simulator_version": "0.1.0-test",
        "payload": base64.b64encode(PbGraphResult().SerializeToString()).decode("ascii"),
    }


def _success_detail() -> dict[str, Any]:
    return {"status": "SUCCESS", "resolved_backend": "single_cpu", "effective_seed": 0, "timeline": {}}


class TestShotResultsAgreeAcrossModes:
    """`shot_results` raises for a graph run in both modes, rather than returning [] in one of them."""

    @patch("claveles.client._simulator_server.requests.request")
    def test_a_graph_run_on_the_simulator_server_has_no_shot_results(self, mock_request: MagicMock) -> None:
        client = SimulatorClient(
            n_shots=1, mode="simulator-server", url="https://css.local", api_key="k", polling_interval=0.0
        )
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, _success_detail()),
            _make_response(HTTPStatus.OK, _graph_envelope()),
            _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
        ]

        result = client.run(GraphRepr(1, 1))

        assert result.graph_result is not None
        with pytest.raises(ValueError, match="only available for circuit execution"):
            _ = result.shot_results

    def test_a_graph_run_locally_has_no_shot_results(self) -> None:
        pytest.importorskip("claveles_simulator")

        result = SimulatorClient(n_shots=1, mode="local").run(GraphRepr(2, 2))

        with pytest.raises(ValueError, match="only available for circuit execution"):
            _ = result.shot_results


class TestAnUnsupportedRepresentationIsRefusedTheSameWay:
    """Every client raises TypeError here, not a ValueError naming `type(representation)`."""

    def test_the_cloud_client_raises_a_type_error(self) -> None:
        client = OptQCClient(url="https://compute.cloud.optqc.com", api_key="k", target="cloud")

        with pytest.raises(TypeError, match="Unsupported representation type: NotARepresentation"):
            client.submit_job(NotARepresentation())  # ty: ignore[invalid-argument-type]

    def test_both_clients_report_it_identically(self) -> None:
        value = NotARepresentation()
        optqc = OptQCClient(url="https://compute.cloud.optqc.com", api_key="k", target="cloud")
        simulator = SimulatorClient(mode="local", n_shots=1)

        with pytest.raises(TypeError) as optqc_error:
            optqc.submit_job(value)  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError) as simulator_error:
            simulator.run(value)  # ty: ignore[invalid-argument-type]

        assert str(optqc_error.value) == str(simulator_error.value)
        assert str(optqc_error.value) == unsupported_representation_message(value)

    def test_it_is_refused_before_anything_is_compiled_or_uploaded(self) -> None:
        # No HTTP mocking: reaching the network would mean the check came too late.
        client = OptQCClient(url="https://compute.cloud.optqc.com", api_key="k", target="cloud")

        with patch.object(client, "_compiler") as compiler, pytest.raises(TypeError):
            client.submit_job(NotARepresentation())  # ty: ignore[invalid-argument-type]

        compiler.circuit_to_graph.assert_not_called()


class TestCredentialsAndTargetAreCheckedAtConstruction:
    """An empty URL or key is refused when the client is built, not at connect time."""

    def test_missing_url_and_api_key_are_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPTQC_CLIENT_URL", raising=False)
        monkeypatch.delenv("OPTQC_CLIENT_API_KEY", raising=False)

        with pytest.raises(ValueError, match="requires url and api_key"):
            OptQCClient(target="cloud")

    def test_a_missing_api_key_alone_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPTQC_CLIENT_API_KEY", raising=False)

        with pytest.raises(ValueError, match="requires url and api_key"):
            OptQCClient(url="https://compute.cloud.optqc.com", target="cloud")

    def test_the_same_omission_is_refused_by_the_simulator_client_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPTQC_SIMULATOR_URL", raising=False)
        monkeypatch.delenv("OPTQC_SIMULATOR_API_KEY", raising=False)

        with pytest.raises(ValueError, match="requires url and api_key"):
            SimulatorClient(mode="simulator-server")

    def test_an_unknown_target_is_named_as_such_even_without_a_qc_node_id(self) -> None:
        # Reporting "qc_node_id is required for target 'Cloud'" would send the reader after the wrong
        # argument: the mistake is the capital C.
        with pytest.raises(ValueError, match=r"Unknown target 'Cloud'"):
            OptQCClient(url="https://compute.cloud.optqc.com", api_key="k", target="Cloud")  # ty: ignore[invalid-argument-type]

    def test_the_accepted_targets_are_the_ones_the_type_names(self) -> None:
        # The annotation is what a reader and a type checker go by, so it has to be the same set the
        # constructor enforces.
        assert set(get_args(OptQCTarget)) == optqc_client._VALID_TARGETS  # noqa: SLF001

    def test_env_var_fallback_still_applies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPTQC_CLIENT_URL", "https://from-env.local")
        monkeypatch.setenv("OPTQC_CLIENT_API_KEY", "env-key")
        monkeypatch.setenv("OPTQC_CLIENT_TARGET", "cloud")

        client = OptQCClient()

        assert client._url == "https://from-env.local"  # noqa: SLF001


class TestJobStatusIsOneVocabulary:
    """The terminal set lists no synonyms, and the two public methods agree on casing."""

    @pytest.mark.parametrize(
        ("wire", "expected"),
        [
            ("SUCCESS", JobStatus.SUCCESS),
            ("success", JobStatus.SUCCESS),
            (" Success ", JobStatus.SUCCESS),
            ("COMPLETED", JobStatus.SUCCESS),
            ("FAILED", JobStatus.FAILURE),
            ("CANCELED", JobStatus.CANCELLED),
            ("CANCELLED", JobStatus.CANCELLED),
            ("SUBMITTED", JobStatus.SUBMITTED),
        ],
    )
    def test_a_state_arrives_under_one_name(self, wire: str, expected: JobStatus) -> None:
        assert JobStatus.from_wire(wire) is expected

    def test_an_unknown_status_is_surfaced_rather_than_waited_out(self) -> None:
        # Treated as "not finished yet", a job that had already stopped would poll until the deadline and
        # then be reported as a timeout.
        with pytest.raises(ClientServerError, match="does not know"):
            JobStatus.from_wire("HALF_DONE")

    def test_which_states_stop_the_polling_loop(self) -> None:
        stopping = {status for status in JobStatus if status.is_terminal}

        # A job leaves SUBMITTED, QUEUED and RUNNING on its own; it leaves none of the others.
        assert stopping == {JobStatus.SUCCESS, JobStatus.FAILURE, JobStatus.TIMEOUT, JobStatus.CANCELLED}

    def test_the_cloud_routes_initial_state_polls_rather_than_raises(self) -> None:
        # A job sits in SUBMITTED until the bridge acknowledges it, so the first poll after a cloud
        # submission reads it. Not knowing the name made every cloud wait end on that first poll.
        assert JobStatus.from_wire("SUBMITTED") is JobStatus.SUBMITTED
        assert not JobStatus.SUBMITTED.is_terminal
        assert not JobStatus.SUBMITTED.is_success

    def test_only_success_is_a_success(self) -> None:
        assert JobStatus.SUCCESS.is_success
        assert not any(s.is_success for s in JobStatus if s is not JobStatus.SUCCESS)

    def test_comparing_against_a_string_still_reads_true(self) -> None:
        assert JobStatus.SUCCESS == "SUCCESS"

    @patch("claveles.client.optqc_common.sleep")
    @patch("claveles.client.optqc_common.requests.request")
    def test_the_two_public_methods_report_the_same_spelling(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,  # noqa: ARG002
    ) -> None:
        client = OptQCClient(
            url="https://scheduler.local",
            api_key="k",
            target="direct",
            qc_node_id="node-001",
            polling_interval=0.0,
        )
        # The server spells it one way, and both methods normalize it, rather than one passing the wire
        # spelling through while the other returns the uppercased form.
        mock_request.return_value = _make_response(HTTPStatus.OK, json_data={"status": "completed"})
        context = OptQCSubmissionContext(job_id="job-1")

        assert client.get_job_status(context) is JobStatus.SUCCESS
        assert client.wait_for_completion(context) is JobStatus.SUCCESS


class TestAFailedJobIsOneExceptionEverywhere:
    """A failed circuit raises the same class locally and on the cloud."""

    @patch("claveles.client._simulator_server.sleep")
    @patch("claveles.client._simulator_server.requests.request")
    def test_the_simulator_server_raises_a_job_failed_error(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,  # noqa: ARG002
    ) -> None:
        client = SimulatorClient(
            n_shots=1, mode="simulator-server", url="https://css.local", api_key="k", polling_interval=0.0
        )
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, {"status": "FAILURE", "error": {"code": "OVERFLOW", "message": "boom"}}),
        ]

        with pytest.raises(JobFailedError) as error:
            client.run(GraphRepr(1, 1))

        assert error.value.args == ("boom", 205)
        assert isinstance(error.value, RuntimeError)

    @patch("claveles.client._simulator_server.sleep")
    @patch("claveles.client._simulator_server.requests.request")
    def test_a_failed_job_is_left_on_the_server_for_inspection(
        self,
        mock_request: MagicMock,
        mock_sleep: MagicMock,  # noqa: ARG002
    ) -> None:
        client = SimulatorClient(
            n_shots=1, mode="simulator-server", url="https://css.local", api_key="k", polling_interval=0.0
        )
        mock_request.side_effect = [
            _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
            _make_response(HTTPStatus.OK, {"status": "FAILURE", "error": {"code": "OVERFLOW", "message": "boom"}}),
        ]

        with pytest.raises(JobFailedError):
            client.run(GraphRepr(1, 1))

        # A DELETE on the failure path would remove the only account of why the job failed.
        assert [call.args[0] for call in mock_request.call_args_list].count("DELETE") == 0

    def test_the_local_simulator_reports_a_failure_the_same_way(self) -> None:
        pytest.importorskip("claveles_simulator")

        client = SimulatorClient(n_shots=1, mode="local")

        with (
            patch(
                "claveles.client._local_simulator_claveles.simulate_graph",
                side_effect=RuntimeError("boom", 205),
            ),
            pytest.raises(JobFailedError) as error,
        ):
            client.run(GraphRepr(2, 2))

        assert error.value.args == ("boom", 205)
        assert error.value.error_id == 205

    def test_a_local_failure_without_an_error_id_still_reads_as_a_job_failure(self) -> None:
        pytest.importorskip("claveles_simulator")

        client = SimulatorClient(n_shots=1, mode="local")

        with (
            patch("claveles.client._local_simulator_claveles.simulate_graph", side_effect=RuntimeError("plain")),
            pytest.raises(JobFailedError) as error,
        ):
            client.run(GraphRepr(2, 2))

        assert error.value.args == ("plain",)
        assert error.value.error_id is None


class TestACircuitStillRunsOnBothClients:
    """A guard that the validation added above did not close the ordinary path."""

    @patch("claveles.client.optqc_common.requests.request")
    def test_the_cloud_client_accepts_a_circuit_and_compiles_it(self, mock_request: MagicMock) -> None:
        client = OptQCClient(
            url="https://scheduler.local",
            api_key="k",
            target="direct",
            qc_node_id="node-001",
            compiler=MagicMock(),
        )
        mock_request.return_value = _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "job-1"})

        with patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}):
            context = client.submit_job(CircuitRepr("ok"))

        assert context.job_id == "job-1"
