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

"""Tests for which representation levels an OptQC result reports.

The machine measures macronodes, so a machinery result is what it reports and the other two levels are
decoded back from it. Decoding one needs the representation it was compiled from, which is what decides
how many levels a result carries: a circuit submission leaves all three behind, a graph submission the
lower two, a machinery submission only its own. The documentation states that matrix
(``docs/source/execution.ipynb``), so it is pinned here rather than left to be read off the decoder.

`SimulatorClient` is the other half of the picture and answers differently -- it runs each
representation as given and decodes nothing, so its result carries the one level that was run. That side
is covered by `test_representation_dispatch.py`.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from claveles.circuit import CircuitRepr, CircuitResult
from claveles.circuit.ops import intrinsic
from claveles.client import OptQCClient
from claveles.graph import GraphRepr, GraphResult
from claveles.graph.constant import BLANK_MODE
from claveles.graph.ops import Initialization, Measurement
from claveles.machinery import MachineryRepr, MachineryResult

if TYPE_CHECKING:
    from claveles.client import OptQCClientResult
    from claveles.client.abstract import ReprType

# The machine's fixed number of local macronodes, which a graph has to match to compile for it.
_N_LOCAL_MACRONODES = 129


def _make_response(status_code: int = 200, json_data: dict[str, Any] | None = None, content: bytes = b"") -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.content = content
    resp.raise_for_status.side_effect = None
    return resp


def _result_content(readout_macronode_indices: list[int]) -> bytes:
    """Build the machinery result JSON the hardware returns, one shot deep.

    The readout macronodes have to be the ones the submitted program asked for: decoding to a graph
    result reads the values back per macronode, so a mismatch decodes into the wrong shape.
    """
    return json.dumps({
        "readout_macronode_indices": readout_macronode_indices,
        "shots": [{"values": [1.0, 2.0, 3.0, 4.0] * len(readout_macronode_indices)}],
    }).encode()


def _circuit() -> CircuitRepr:
    circuit = CircuitRepr("levels")
    _ = circuit.Q(0) | intrinsic.PhaseRotation(0.1)
    _ = circuit.Q(0) | intrinsic.Measurement(0.0)
    return circuit


def _graph() -> GraphRepr:
    graph = GraphRepr(_N_LOCAL_MACRONODES, 2)
    graph.place_operation(Initialization((0, 0), 0.0, (BLANK_MODE, 0)))
    graph.place_operation(Measurement((0, 1), 0.0, readout=True))
    return graph


def _machinery() -> MachineryRepr:
    return MachineryRepr.from_graph_repr(_graph())


def _run_against_a_mocked_scheduler(representation: ReprType) -> OptQCClientResult:
    """Submit a representation and retrieve its result, with the scheduler mocked out."""
    client = OptQCClient(url="https://scheduler.local", api_key="k", target="direct", qc_node_id="node-001", n_shots=1)

    with (
        patch("claveles.client.optqc_common.requests.request") as mock_request,
        # The machinery JSON on the wire has its own tests; what matters here is which representations
        # the submission leaves behind for the decoder.
        patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}),
    ):
        mock_request.side_effect = [_make_response(HTTPStatus.ACCEPTED, {"job_id": "job-001"})]
        context = client.submit_job(representation)

        assert context.machinery_repr is not None  # every accepted representation lowers to machinery
        mock_request.side_effect = [
            _make_response(HTTPStatus.OK, {"status": "SUCCESS"}),
            _make_response(HTTPStatus.OK, content=_result_content(context.machinery_repr.readout_macronode_indices)),
        ]
        return client.get_job_result(context)


def _levels_reported(result: OptQCClientResult) -> dict[str, bool]:
    return {
        "circuit": result.circuit_result is not None,
        "graph": result.graph_result is not None,
        "machinery": result.machinery_result is not None,
    }


class TestASubmissionDecidesWhichLevelsAreReported:
    @pytest.mark.parametrize(
        ("submitted", "levels"),
        [
            pytest.param(
                _circuit(),
                {"circuit": True, "graph": True, "machinery": True},
                id="a circuit fills all three",
            ),
            pytest.param(
                _graph(),
                {"circuit": False, "graph": True, "machinery": True},
                id="a graph fills the lower two",
            ),
            pytest.param(
                _machinery(),
                {"circuit": False, "graph": False, "machinery": True},
                id="a machinery representation fills only its own",
            ),
        ],
    )
    def test_the_levels_reported_are_the_submitted_one_and_below(
        self, submitted: ReprType, levels: dict[str, bool]
    ) -> None:
        # Machinery is `True` in every row because it is the level the machine reports: no submission
        # can leave it out, and the other two are decoded from it.
        assert _levels_reported(_run_against_a_mocked_scheduler(submitted)) == levels

    @pytest.mark.parametrize(
        ("submitted", "highest"),
        [
            pytest.param(_circuit(), CircuitResult, id="circuit"),
            pytest.param(_graph(), GraphResult, id="graph"),
            pytest.param(_machinery(), MachineryResult, id="machinery"),
        ],
    )
    def test_execution_result_is_the_level_that_was_submitted(self, submitted: ReprType, highest: type) -> None:
        # `execution_result` picks the highest level available, which is the one that was submitted.
        assert isinstance(_run_against_a_mocked_scheduler(submitted).execution_result, highest)
