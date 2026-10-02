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

"""Tests for turning a graph-representation result back into per-mode circuit measurements.

This is the conversion `claveles.client.optqc_common` performs on the way back from the hardware, which
reports what each macronode measured while the caller asked about circuit modes. What these pin is the
mapping from a macronode's b and d micronodes to the modes leaving it, and the skipping of every
macronode that is not a measurement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic
from claveles.circuit.result import CircuitResult
from claveles.graph.convert import GreedyConverter, GreedyConvertSettings
from claveles.graph.result import GraphMacronodeMeasuredValue, GraphResult, GraphShotMeasuredValue

if TYPE_CHECKING:
    from claveles.graph.program import GraphRepr

_N_MODES = 3


def _circuit_and_graph() -> tuple[CircuitRepr, GraphRepr]:
    """Build a small circuit that measures every mode, and its graph form."""
    # The modes are left open: what a mode starts from has no bearing on how the measured values come
    # back, and a graph carries the machine's own squeezed state on every mode regardless.
    circuit = CircuitRepr("result_from_graph")
    circuit.Q(0, 1) | intrinsic.ControlledZ(0.0)
    for mode in range(_N_MODES):
        circuit.Q(mode) | intrinsic.Measurement(0.0)

    graph = GreedyConverter(GreedyConvertSettings(n_local_macronodes=4)).convert(circuit)
    return circuit, graph


def _graph_result_over(graph: GraphRepr, n_shots: int) -> GraphResult:
    """Fill every macronode of `graph` with values derived from its index, for `n_shots` shots."""
    shots = []
    for shot in range(n_shots):
        values = []
        for index in range(graph.n_total_macronodes):
            h, w = graph.get_coord(index)
            values.append(
                GraphMacronodeMeasuredValue(
                    index,
                    h,
                    w,
                    m_b=float(100 * shot + index),
                    m_d=-float(100 * shot + index),
                ),
            )
        shots.append(GraphShotMeasuredValue(values, n_local_macronodes=graph.n_local_macronodes))
    return GraphResult(graph.n_local_macronodes, shots)


def test_every_shot_is_carried_over() -> None:
    _, graph = _circuit_and_graph()

    result = CircuitResult.construct_from_graph_result(_graph_result_over(graph, 3), graph)

    assert result.n_shots == 3
    assert len(result) == 3


def test_each_shot_reports_one_value_per_measured_mode() -> None:
    _, graph = _circuit_and_graph()

    result = CircuitResult.construct_from_graph_result(_graph_result_over(graph, 2), graph)

    for shot in result:
        assert {mmv.mode for mmv in shot} == set(range(_N_MODES))


def test_the_modes_come_out_in_order() -> None:
    # The values are gathered into a dict keyed by mode, so the ordering is the conversion's own doing.
    _, graph = _circuit_and_graph()

    result = CircuitResult.construct_from_graph_result(_graph_result_over(graph, 1), graph)

    modes = [mmv.mode for mmv in result[0]]
    assert modes == sorted(modes)


def test_each_value_comes_from_the_macronode_that_measured_that_mode() -> None:
    _, graph = _circuit_and_graph()

    result = CircuitResult.construct_from_graph_result(_graph_result_over(graph, 1), graph)

    # Rebuild the expected mapping the way the hardware result describes it: for every measurement
    # macronode, the b micronode reports the mode leaving to the right and d the one leaving upward.
    expected: dict[int, float] = {}
    for index in range(graph.n_total_macronodes):
        h, w = graph.get_coord(index)
        if not graph.is_measurement(h, w):
            continue
        mode_d, mode_b = graph.calc_io_of_macronode(h, w)[0], graph.calc_io_of_macronode(h, w)[1]
        if mode_b >= 0:
            expected[mode_b] = float(index)
        if mode_d >= 0:
            expected[mode_d] = -float(index)

    assert {mmv.mode: mmv.value for mmv in result[0]} == pytest.approx(expected)


def test_a_graph_result_of_no_shots_gives_no_shots() -> None:
    _, graph = _circuit_and_graph()

    result = CircuitResult.construct_from_graph_result(GraphResult(graph.n_local_macronodes, []), graph)

    assert result.n_shots == 0


def test_indexing_and_slicing_reach_the_shots() -> None:
    _, graph = _circuit_and_graph()

    result = CircuitResult.construct_from_graph_result(_graph_result_over(graph, 3), graph)

    assert result[0] is not None
    assert len(result[0:2]) == 2
