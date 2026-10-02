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

"""Tests for how SimulatorClient rejects a value that is none of the three representations.

Both modes raise `TypeError` at the point the value is handed over. Without an `else` on `mode="local"`'s
dispatch the simulator would never be called, and a result holding none of the three would come back as
though the run had succeeded -- surfacing only later, as an empty-message `ValueError` out of
`execution_result`, and only in that one mode.
"""

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import BosonicState
from claveles.client import SimulatorClient
from claveles.client.abstract import unsupported_representation_message
from claveles.client.simulator_client import ExecutionDetails, SimulatorClientResult
from claveles.graph import GraphRepr


class NotARepresentation:
    """Stand-in for whatever a caller passes by mistake."""


def test_local_mode_rejects_an_unsupported_representation():
    client = SimulatorClient(mode="local", n_shots=1)

    with pytest.raises(TypeError, match="Unsupported representation type: NotARepresentation"):
        client.run(NotARepresentation())  # ty: ignore[invalid-argument-type]


def test_both_modes_report_the_same_misuse_identically():
    # The rejection happens before anything is uploaded, so the server mode needs no HTTP mocking.
    value = NotARepresentation()
    local = SimulatorClient(mode="local", n_shots=1)
    server = SimulatorClient(mode="simulator-server", n_shots=1, url="https://css.local", api_key="k")

    with pytest.raises(TypeError) as local_error:
        local.run(value)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError) as server_error:
        server.run(value)  # ty: ignore[invalid-argument-type]

    assert str(local_error.value) == str(server_error.value)


def test_rejection_names_the_type_and_the_expected_ones():
    message = unsupported_representation_message(NotARepresentation())

    assert "NotARepresentation" in message
    for expected in ("CircuitRepr", "GraphRepr", "MachineryRepr"):
        assert expected in message


def test_a_result_holding_no_result_at_all_cannot_be_built():
    # A run produces the one result of the representation it was given, and `SimulatorClientResult`
    # now holds exactly that one. There is no longer a "none of the three was set" state for
    # `execution_result` to have to report, so the field is required rather than three optional ones.
    with pytest.raises(TypeError, match="_result"):
        SimulatorClientResult(  # ty: ignore[missing-argument]
            execution_details=ExecutionDetails(scheduler_version="", simulator_version=""),
            _shot_results=None,
        )


def test_asking_a_graph_run_for_a_circuit_result_says_what_ran():
    pytest.importorskip("claveles_simulator")

    result = SimulatorClient(mode="local", n_shots=1).run(GraphRepr(2, 2))

    with pytest.raises(ValueError, match="this run was a graph") as error:
        _ = result.circuit_result

    # The accessor a caller should read instead is named, rather than only what is missing.
    assert "graph_result" in str(error.value)


def test_a_supported_representation_still_runs():
    pytest.importorskip("claveles_simulator")

    circuit = CircuitRepr("supported")
    _ = circuit.Q(0) | BosonicState.vacuum()
    _ = circuit.Q(0) | std.MeasureHomodyne(0.0)

    result = SimulatorClient(mode="local", n_shots=2, backend="cpu").run(circuit)

    assert result.n_shots == 2
