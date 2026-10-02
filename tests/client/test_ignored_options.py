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

"""Tests that an option which will not be read is reported rather than dropped in silence.

`SimulatorClient` takes every option whatever it is asked to run, and reads only the ones that apply.
Dropping the rest silently would let `track_truncation_error=True` on a graph run return
`truncation_error=None` and read as "nothing was truncated". The state-saving options are checked on the
same footing: a mode that cannot honour them refuses every value it was given rather than only the ones
that *differ* from the default, and the check runs on the attributes, so a policy assigned after
construction is refused too.

The graph and machinery runs here go through `mode="simulator-server"` with the HTTP layer mocked, so
they check a whole `run()` call without needing the optional simulator wheel.
"""

from __future__ import annotations

import base64
import warnings
from http import HTTPStatus
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import BosonicState
from claveles.client import SimulatorClient
from claveles.client.simulator_client import (
    DEFAULT_MAX_SAVED_STATE_ELEMENTS,
    DEFAULT_RESOURCE_SQUEEZING_LEVEL,
    SIMULATOR_SERVER_STATE_SAVE_POLICY,
)
from claveles.graph import GraphRepr
from claveles.machinery import MachineryRepr
from claveles.pb.claveles_cloud.program.v1.graph_pb2 import GraphResult as PbGraphResult
from claveles.pb.claveles_cloud.program.v1.machinery_pb2 import MachineryResult as PbMachineryResult
from claveles.pb.claveles_cloud.program.v1.quantum_program_pb2 import QuantumProgramResult
from claveles.warnings import IgnoredOptionWarning


def _circuit() -> CircuitRepr:
    circuit = CircuitRepr("options")
    circuit.Q(0) | BosonicState.vacuum()
    circuit.Q(0) | std.MeasureHomodyne(0.0)
    return circuit


def _make_response(status_code: int = 200, json_data: dict[str, Any] | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = "" if json_data is None else str(json_data)
    resp.headers = {}
    return resp


def _envelope(program_type: str) -> dict[str, Any]:
    """A ``ResultEnvelope``-shaped dict carrying an empty result of the given type."""
    payload = {
        "CIRCUIT": QuantumProgramResult,
        "GRAPH": PbGraphResult,
        "MACHINERY": PbMachineryResult,
    }[program_type]()
    return {
        "program_type": program_type,
        "simulator_version": "0.1.0-test",
        "payload": base64.b64encode(payload.SerializeToString()).decode("ascii"),
    }


def _server_responses(program_type: str) -> list[MagicMock]:
    """The submit / poll / fetch / release sequence of a successful job."""
    return [
        _make_response(HTTPStatus.ACCEPTED, {"job_id": "j1", "status": "QUEUED", "created_at": "t"}),
        _make_response(
            HTTPStatus.OK,
            {"status": "SUCCESS", "resolved_backend": "single_cpu", "effective_seed": 0, "timeline": {}},
        ),
        _make_response(HTTPStatus.OK, _envelope(program_type)),
        _make_response(HTTPStatus.NO_CONTENT),  # best-effort DELETE
    ]


def _server_client(**kwargs: Any) -> SimulatorClient:  # noqa: ANN401  # forwards whichever option is under test
    return SimulatorClient(
        n_shots=1,
        mode="simulator-server",
        url="https://css.local",
        api_key="k",
        polling_interval=0.0,
        **kwargs,
    )


class TestCircuitOnlyOptionsOnAGraphRun:
    """The five circuit-only options whose inert value is the same in every mode.

    The other two, `state_save_policy` and `max_saved_state_elements`, only `mode="local"` lets a caller
    choose, so they are covered by `TestTheStateSavingOptionsOnAGraphRun` against that mode.
    """

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("max_bond_dim", 16),
            ("svd_cutoff", 1e-8),
            ("track_truncation_error", True),
            ("state_representation", "fock"),
            ("decompose_std", True),
        ],
    )
    @patch("claveles.client._simulator_server.requests.request")
    def test_setting_one_is_reported(self, mock_request: MagicMock, option: str, value: Any) -> None:  # noqa: ANN401
        mock_request.side_effect = _server_responses("GRAPH")
        client = _server_client(**{option: value})

        with pytest.warns(IgnoredOptionWarning, match=option):
            client.run(GraphRepr(1, 1))

    @patch("claveles.client._simulator_server.requests.request")
    def test_the_message_names_every_option_that_was_dropped(self, mock_request: MagicMock) -> None:
        mock_request.side_effect = _server_responses("GRAPH")
        client = _server_client(max_bond_dim=16, track_truncation_error=True)

        with pytest.warns(IgnoredOptionWarning) as record:
            client.run(GraphRepr(1, 1))

        message = str(record[0].message)
        assert "max_bond_dim" in message
        assert "track_truncation_error" in message

    @patch("claveles.client._simulator_server.requests.request")
    def test_a_machinery_run_reports_them_too(self, mock_request: MagicMock) -> None:
        mock_request.side_effect = _server_responses("MACHINERY")
        client = _server_client(svd_cutoff=1e-8)

        with pytest.warns(IgnoredOptionWarning, match="svd_cutoff"):
            client.run(MachineryRepr(1, 1))

    @patch("claveles.client._simulator_server.requests.request")
    def test_leaving_them_at_the_value_that_asks_for_nothing_is_silent(self, mock_request: MagicMock) -> None:
        # svd_cutoff=0.0 disables truncation and state_representation="auto" leaves the choice open, so
        # passing either asks for nothing on a graph run and gets it.
        mock_request.side_effect = _server_responses("GRAPH")
        client = _server_client(svd_cutoff=0.0, state_representation="auto", max_bond_dim=None)

        with warnings.catch_warnings():
            warnings.simplefilter("error", IgnoredOptionWarning)
            client.run(GraphRepr(1, 1))

    @patch("claveles.client._simulator_server.requests.request")
    def test_a_circuit_run_reads_them_and_says_nothing(self, mock_request: MagicMock) -> None:
        mock_request.side_effect = _server_responses("CIRCUIT")
        client = _server_client(max_bond_dim=16, track_truncation_error=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error", IgnoredOptionWarning)
            client.run(_circuit())


class TestTheResourceSqueezingLevelOnACircuitRun:
    """The one option only a graph or machinery program reads."""

    @patch("claveles.client._simulator_server.requests.request")
    def test_setting_it_for_a_circuit_is_reported(self, mock_request: MagicMock) -> None:
        mock_request.side_effect = _server_responses("CIRCUIT")
        client = _server_client(resource_squeezing_level=8.0)

        with pytest.warns(IgnoredOptionWarning, match="resource_squeezing_level"):
            client.run(_circuit())

    @patch("claveles.client._simulator_server.requests.request")
    def test_setting_it_to_what_the_default_resolves_to_is_still_reported(self, mock_request: MagicMock) -> None:
        # Defaulting to this number would make "differs from the default" unable to tell a chosen level
        # from an unchosen one. None is the unchosen value, and choosing a level is choosing it.
        mock_request.side_effect = _server_responses("CIRCUIT")
        client = _server_client(resource_squeezing_level=DEFAULT_RESOURCE_SQUEEZING_LEVEL)

        with pytest.warns(IgnoredOptionWarning, match="resource_squeezing_level"):
            client.run(_circuit())

    @patch("claveles.client._simulator_server.requests.request")
    def test_leaving_it_unchosen_is_silent(self, mock_request: MagicMock) -> None:
        mock_request.side_effect = _server_responses("CIRCUIT")
        client = _server_client()

        assert client.resource_squeezing_level is None
        with warnings.catch_warnings():
            warnings.simplefilter("error", IgnoredOptionWarning)
            client.run(_circuit())

    @patch("claveles.client._simulator_server.requests.request")
    def test_an_unchosen_level_still_reaches_a_graph_run_as_the_default(self, mock_request: MagicMock) -> None:
        mock_request.side_effect = _server_responses("GRAPH")
        client = _server_client()

        client.run(GraphRepr(1, 1))

        # The attribute keeps saying "unchosen"; the resolved level is what goes on the wire.
        assert client.resource_squeezing_level is None
        body = mock_request.call_args_list[0].kwargs["json"]
        assert body["squeezing_level"] == pytest.approx(DEFAULT_RESOURCE_SQUEEZING_LEVEL)

    @patch("claveles.client._simulator_server.requests.request")
    def test_a_chosen_level_reaches_a_graph_run_unchanged(self, mock_request: MagicMock) -> None:
        mock_request.side_effect = _server_responses("GRAPH")
        client = _server_client(resource_squeezing_level=8.0)

        client.run(GraphRepr(1, 1))

        body = mock_request.call_args_list[0].kwargs["json"]
        assert body["squeezing_level"] == pytest.approx(8.0)

    @patch("claveles.client._simulator_server.requests.request")
    def test_the_warning_can_be_promoted_to_an_error(self, mock_request: MagicMock) -> None:
        mock_request.side_effect = _server_responses("CIRCUIT")
        client = _server_client(resource_squeezing_level=8.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error", IgnoredOptionWarning)
            with pytest.raises(IgnoredOptionWarning):
                client.run(_circuit())


class TestTheStateSavingOptionsAreModelledPerMode:
    """`state_save_policy` and `max_saved_state_elements` now report what the mode will do."""

    def test_an_unchosen_policy_reads_as_what_local_mode_applies(self) -> None:
        assert SimulatorClient(n_shots=1).state_save_policy == "none"

    def test_an_unchosen_policy_reads_as_what_the_server_applies(self) -> None:
        # Reading as None in both modes while the two behave differently would let the same code produce
        # a saved state on one and not on the other with nothing to explain it.
        assert _server_client().state_save_policy == SIMULATOR_SERVER_STATE_SAVE_POLICY

    def test_a_chosen_policy_is_kept_for_local_mode(self) -> None:
        assert SimulatorClient(n_shots=1, state_save_policy="all").state_save_policy == "all"

    def test_the_server_refuses_a_chosen_policy(self) -> None:
        with pytest.raises(ValueError, match="state_save_policy cannot be set"):
            _server_client(state_save_policy="all")

    def test_local_mode_defaults_the_saved_state_bound(self) -> None:
        assert SimulatorClient(n_shots=1).max_saved_state_elements == DEFAULT_MAX_SAVED_STATE_ELEMENTS

    def test_local_mode_still_lets_the_bound_be_lifted(self) -> None:
        assert SimulatorClient(n_shots=1, max_saved_state_elements=None).max_saved_state_elements is None

    def test_the_server_asks_for_no_bound_of_its_own(self) -> None:
        assert _server_client().max_saved_state_elements is None

    @pytest.mark.parametrize("value", [None, DEFAULT_MAX_SAVED_STATE_ELEMENTS, 4096])
    def test_the_server_refuses_any_bound_it_was_given(self, value: int | None) -> None:
        # Accepting the default and ignoring it, while refusing None -- the documented way to lift the
        # bound -- would invert the rule. Either the mode can honour the option or it cannot.
        with pytest.raises(ValueError, match="max_saved_state_elements cannot be set"):
            _server_client(max_saved_state_elements=value)


class TestTheStateSavingOptionsOnAGraphRun:
    """The two circuit-only options a caller can choose only under ``mode="local"``.

    Only a circuit simulation hands states back, so a graph or machinery run reads neither. They sit
    outside the table the class above parametrizes because their inert value depends on the mode: the
    server modes write the policy and bound CSS applies into them, so under those modes there is no
    caller choice to report.
    """

    @staticmethod
    def _local_client(**kwargs: Any) -> SimulatorClient:  # noqa: ANN401  # forwards whichever option is under test
        pytest.importorskip("claveles_simulator")
        return SimulatorClient(n_shots=1, mode="local", backend="cpu", **kwargs)

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("state_save_policy", "all"),
            ("max_saved_state_elements", 4096),
        ],
    )
    def test_setting_one_is_reported(self, option: str, value: Any) -> None:  # noqa: ANN401
        client = self._local_client(**{option: value})

        with pytest.warns(IgnoredOptionWarning, match=option):
            client.run(GraphRepr(2, 2))

    def test_the_message_names_both_when_both_were_set(self) -> None:
        client = self._local_client(state_save_policy="all", max_saved_state_elements=4096)

        with pytest.warns(IgnoredOptionWarning) as record:
            client.run(GraphRepr(2, 2))

        message = str(record[0].message)
        assert "state_save_policy" in message
        assert "max_saved_state_elements" in message

    def test_a_machinery_run_reports_them_too(self) -> None:
        client = self._local_client(state_save_policy="all")

        with pytest.warns(IgnoredOptionWarning, match="state_save_policy"):
            client.run(MachineryRepr(2, 2))

    def test_leaving_them_at_what_local_mode_applies_is_silent(self) -> None:
        client = self._local_client(
            state_save_policy="none", max_saved_state_elements=DEFAULT_MAX_SAVED_STATE_ELEMENTS
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", IgnoredOptionWarning)
            client.run(GraphRepr(2, 2))

    def test_a_circuit_run_reads_them_and_says_nothing(self) -> None:
        client = self._local_client(state_save_policy="all", max_saved_state_elements=4096)

        with warnings.catch_warnings():
            warnings.simplefilter("error", IgnoredOptionWarning)
            client.run(_circuit())

    @patch("claveles.client._simulator_server.requests.request")
    def test_the_server_modes_report_nothing_the_caller_did_not_choose(self, mock_request: MagicMock) -> None:
        # The server modes hold `first_only` and `None`, neither of which a caller can set, so a graph
        # run under them has no dropped choice to report -- unlike a `local` run holding the same
        # policy, which a caller did choose.
        mock_request.side_effect = _server_responses("GRAPH")
        client = _server_client()
        assert client.state_save_policy == SIMULATOR_SERVER_STATE_SAVE_POLICY

        with warnings.catch_warnings():
            warnings.simplefilter("error", IgnoredOptionWarning)
            client.run(GraphRepr(1, 1))


class TestAnOptionAssignedAfterConstruction:
    """The attributes are public and documented as assignable, so the check cannot stop at __init__."""

    def test_the_server_refuses_a_policy_assigned_afterwards(self) -> None:
        client = _server_client()
        client.state_save_policy = "all"

        # Accepting it here would mean ignoring it: the server path has "first_only" written into it.
        with pytest.raises(ValueError, match="mode='simulator-server' always applies"):
            client.run(GraphRepr(1, 1))

    def test_the_server_refuses_a_bound_assigned_afterwards(self) -> None:
        client = _server_client()
        client.max_saved_state_elements = 4096

        with pytest.raises(ValueError, match="bound of the server's own"):
            client.run(GraphRepr(1, 1))

    def test_it_is_refused_before_anything_is_uploaded(self) -> None:
        # No HTTP mocking: reaching the network would mean the check came too late.
        client = _server_client()
        client.state_save_policy = "all"

        with pytest.raises(ValueError, match="mode='simulator-server' always applies"):
            client.run(GraphRepr(1, 1))

    def test_local_mode_honours_a_policy_assigned_afterwards(self) -> None:
        pytest.importorskip("claveles_simulator")

        client = SimulatorClient(n_shots=2)
        client.state_save_policy = "all"

        result = client.run(_circuit())

        assert all(shot.state is not None for shot in result.shot_results)
