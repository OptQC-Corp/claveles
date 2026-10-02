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

"""Tests for the contract AbstractClientResult imposes on every client result.

`execution_time` is an abstract property rather than a bare annotation: an annotation creates no class
attribute and so leaves nothing for `ABC` to enforce, and a result that never set it would raise
`AttributeError` from `execute(...).execution_time` instead of returning a value.
"""

import json
from http import HTTPStatus
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
import requests

from claveles.circuit.result import CircuitResult, CircuitShotMeasuredValue
from claveles.client import OptQCClient, OptQCClientResult
from claveles.client.abstract import AbstractClientResult, ResultType
from claveles.execute import execute


class DummyMachineryRepr:
    """Stand-in for MachineryRepr, patched in so no real compilation is needed."""

    n_local_macronodes: int = 129
    n_steps: int = 1
    readout_macronode_indices: ClassVar[list[int]] = [0]


def _make_response(status_code: int = 200, json_data: dict[str, Any] | None = None, content: bytes = b"") -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.content = content
    resp.raise_for_status.side_effect = requests.HTTPError(response=resp) if status_code >= 400 else None
    return resp


def _make_result_content() -> bytes:
    return json.dumps({
        # Hardware returns the singular key (see json_to_machinery_result).
        "readout_macronode_indices": [0],
        "shots": [{"values": [1.0, 2.0, 3.0, 4.0]}],
    }).encode()


def test_result_without_execution_time_cannot_be_instantiated():
    class MissingExecutionTime(AbstractClientResult):
        @property
        def execution_result(self) -> ResultType:
            return CircuitResult([])

    with pytest.raises(TypeError, match="execution_time"):
        MissingExecutionTime()


def test_result_without_execution_result_cannot_be_instantiated():
    class MissingExecutionResult(AbstractClientResult):
        @property
        def execution_time(self) -> None:
            return None

    with pytest.raises(TypeError, match="execution_result"):
        MissingExecutionResult()


def test_optqc_result_reports_no_execution_time():
    # The job endpoints carry a status and a result location, and no timing.
    assert OptQCClientResult(circuit_result=CircuitResult([])).execution_time is None


def test_n_shots_is_read_off_the_execution_result():
    circuit_result = CircuitResult([CircuitShotMeasuredValue(), CircuitShotMeasuredValue()])

    # Inherited from the base rather than implemented per client result.
    assert OptQCClientResult(circuit_result=circuit_result).n_shots == 2


@patch("claveles.client.optqc_common.sleep")
# Both, because the client checks what it was handed in one place (`abstract.representation_kind`) and
# then branches on it in another (`submit_job`'s lowering to machinery).
@patch("claveles.client.abstract.MachineryRepr", DummyMachineryRepr)
@patch("claveles.client.optqc_client.MachineryRepr", DummyMachineryRepr)
@patch("claveles.client.optqc_common.requests.request")
def test_execution_time_is_readable_on_the_optqc_route(
    mock_request: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """`execution_time` reads as None on the OptQC route, which reports no backend timing."""
    mock_request.side_effect = [
        _make_response(HTTPStatus.ACCEPTED, json_data={"job_id": "job-1"}),
        _make_response(HTTPStatus.OK, json_data={"status": "SUCCESS"}),
        # get_job_result reads the status again before downloading, as the cloud route does.
        _make_response(HTTPStatus.OK, json_data={"status": "SUCCESS"}),
        _make_response(HTTPStatus.OK, content=_make_result_content()),
    ]
    client = OptQCClient(
        url="https://scheduler.local",
        api_key="test-api-key",
        target="direct",
        qc_node_id="node-001",
        n_shots=1,
    )

    with patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}):
        result = execute(DummyMachineryRepr(), client)  # ty: ignore[invalid-argument-type]

    assert result.execution_time is None
    # total_time is measured on this side, so it is available whatever the backend reports.
    assert result.total_time is not None
    assert mock_sleep.call_count == 0
