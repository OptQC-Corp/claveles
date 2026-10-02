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

"""Tests for reading a composite machinery run back into one result per composed shot.

`claveles.machinery.compose` is the machinery-side twin of `claveles.graph.compose`'s decomposition: a graph
is composed, converted to machinery, run, and the machinery result is then split back up. Nothing else
in the repository references the module, so these tests are what says what it promises: the regrouping
by shot, and the dropping of macronodes that belong to no shot.
"""

from __future__ import annotations

import pytest

from claveles.graph import GraphRepr
from claveles.graph.compose import ComposeSettings, MappingInfo, compose_into_composite_graph
from claveles.graph.ops import PhaseRotation
from claveles.machinery.compose import decompose_composite_machinery_result
from claveles.machinery.result import (
    MachineryMacronodeMeasuredValue,
    MachineryResult,
    MachineryShotMeasuredValue,
)


def _composite_of(graph: GraphRepr, n_shots: int) -> tuple[int, MappingInfo]:
    """Compose `graph` `n_shots` times and report the composite size and the mapping.

    Args:
        graph: The graph to compose.
        n_shots: How many shots to compose into one graph.

    Returns:
        tuple[int, MappingInfo]: The composite macronode count and the mapping back to the original.
    """
    composite = compose_into_composite_graph(
        graph,
        ComposeSettings(n_shots=n_shots, n_local_macronodes=5, max_steps=5),
    )
    return composite.graph.n_total_macronodes, composite.mapping


def _one_shot_over(n_macronodes: int) -> MachineryResult:
    """Build a machinery result of one shot covering `n_macronodes`, with values read off the index.

    Args:
        n_macronodes: How many macronodes the composite machinery run covered.

    Returns:
        MachineryResult: The composite result.
    """
    values = [
        MachineryMacronodeMeasuredValue(
            float(index),
            float(index) + 0.25,
            float(index) + 0.5,
            float(index) + 0.75,
            index=index,
        )
        for index in range(n_macronodes)
    ]
    return MachineryResult([MachineryShotMeasuredValue(values)])


def _graph_of_one_operation() -> GraphRepr:
    graph = GraphRepr(1, 1)
    graph.place_operation(PhaseRotation(macronode=(0, 0), phi=0.0, swap=False))
    return graph


@pytest.mark.parametrize("n_shots", [1, 3])
def test_one_composite_shot_becomes_one_result_per_composed_shot(n_shots: int) -> None:
    n_macronodes, mapping = _composite_of(_graph_of_one_operation(), n_shots)

    decomposed = decompose_composite_machinery_result(_one_shot_over(n_macronodes), mapping)

    assert len(decomposed.measured_vals) == n_shots


def test_all_four_micronodes_are_carried_to_the_macronode_they_came_from() -> None:
    n_macronodes, mapping = _composite_of(_graph_of_one_operation(), 3)

    decomposed = decompose_composite_machinery_result(_one_shot_over(n_macronodes), mapping)

    for composite_index, (i_shot, original_index) in mapping.map.items():
        mmv = decomposed.measured_vals[i_shot].items[original_index]
        assert mmv.index == original_index
        assert mmv.m_a == pytest.approx(float(composite_index))
        assert mmv.m_b == pytest.approx(float(composite_index) + 0.25)
        assert mmv.m_c == pytest.approx(float(composite_index) + 0.5)
        assert mmv.m_d == pytest.approx(float(composite_index) + 0.75)


def test_a_macronode_outside_the_mapping_is_dropped() -> None:
    # The composite graph is padded out to its step count, and the padding belongs to no shot.
    n_macronodes, mapping = _composite_of(_graph_of_one_operation(), 3)
    assert n_macronodes > len(mapping.map), "expected padding to exist"

    decomposed = decompose_composite_machinery_result(_one_shot_over(n_macronodes), mapping)

    assert sum(len(smv.items) for smv in decomposed) == len(mapping.map)


def test_several_composite_shots_each_expand_into_their_own_results() -> None:
    # measured_vals is extended per composite shot, so two of them give twice as many results.
    n_macronodes, mapping = _composite_of(_graph_of_one_operation(), 2)
    one = _one_shot_over(n_macronodes).measured_vals[0]
    composite = MachineryResult([one, one])

    decomposed = decompose_composite_machinery_result(composite, mapping)

    assert len(decomposed.measured_vals) == 2 * 2


def test_a_composite_result_of_no_shots_decomposes_to_no_shots() -> None:
    _, mapping = _composite_of(_graph_of_one_operation(), 2)

    decomposed = decompose_composite_machinery_result(MachineryResult([]), mapping)

    assert len(decomposed.measured_vals) == 0
