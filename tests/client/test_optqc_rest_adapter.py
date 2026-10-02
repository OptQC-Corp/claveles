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

"""Tests for the OptQC REST adapter (serialization/deserialization)."""

from math import pi
from unittest.mock import MagicMock

import pytest

from claveles.client import OptQCClientError, json_to_machinery_result, machinery_repr_to_json
from claveles.feedforward import FeedForward, Variable
from claveles.graph import GraphRepr
from claveles.graph.constant import BLANK_MODE
from claveles.graph.ops import Initialization, Measurement, PhaseRotation
from claveles.machinery.program import FFCoeffMatrixGenerationMethods, MachineryRepr


class _DummyVariable(Variable[str]):
    """Minimal feedforward variable for adapter tests."""

    def get_from_operation(self) -> str:
        return "dummy_op"


def _create_mock_machinery_repr(
    n_local: int = 2, n_steps: int = 1, n_readout_indices: list[int] | None = None
) -> MagicMock:
    """Create a minimal mock MachineryRepr for testing."""
    n_total = n_local * n_steps
    repr_mock = MagicMock()
    repr_mock.n_local_macronodes = n_local
    repr_mock.n_steps = n_steps

    angle_mock = MagicMock()
    angle_mock.theta_a = 1.0
    angle_mock.theta_b = 2.0
    angle_mock.theta_c = 3.0
    angle_mock.theta_d = 4.0
    repr_mock.get_homodyne_angle.return_value = angle_mock

    repr_mock.get_displacement_k_minus_1.return_value = (0.1, 0.2)
    repr_mock.get_displacement_k_minus_n.return_value = (0.3, 0.4)

    repr_mock.ff_coeff_matrix_k_plus_1.generation_methods = [
        FFCoeffMatrixGenerationMethods.FROM_MACRONODE_ANGLE
    ] * n_total
    repr_mock.ff_coeff_matrix_k_plus_n.generation_methods = [FFCoeffMatrixGenerationMethods.ZERO_FILLED] * n_total

    repr_mock.readout_macronode_indices = n_readout_indices or [0]

    return repr_mock


def _create_reference_graph() -> GraphRepr:
    """Create the reference graph: initialization, phase rotation, measurement."""
    graph = GraphRepr(2, 3)
    graph.place_operation(Initialization((0, 0), 0.0, (BLANK_MODE, 0)))
    graph.place_operation(PhaseRotation((0, 1), pi / 4, swap=False))
    graph.place_operation(Measurement((0, 2), pi / 2, readout=True))
    return graph


class TestMachineryReprToJson:
    """Test machinery_repr_to_json conversion."""

    def test_basic_conversion(self) -> None:
        """Test basic conversion to JSON payload."""
        mock_repr = _create_mock_machinery_repr(n_local=2, n_steps=1)
        result = machinery_repr_to_json(mock_repr)

        assert result["n_local_macronodes"] == 2
        assert result["n_steps"] == 1
        assert len(result["homodyne_angles"]) == 2 * 1 * 4  # n_total * 4 angles
        assert result["homodyne_angles"][:4] == [1.0, 2.0, 3.0, 4.0]

    def test_displacements(self) -> None:
        """Test displacement extraction."""
        mock_repr = _create_mock_machinery_repr(n_local=2, n_steps=1)
        result = machinery_repr_to_json(mock_repr)

        assert len(result["displacements_k_minus_1"]) == 2 * 1 * 2
        assert result["displacements_k_minus_1"][:2] == [0.1, 0.2]
        assert result["displacements_k_minus_n"][:2] == [0.3, 0.4]

    def test_generation_methods(self) -> None:
        """Test generation method mapping."""
        mock_repr = _create_mock_machinery_repr(n_local=2, n_steps=1)
        result = machinery_repr_to_json(mock_repr)

        assert result["generating_method_for_ff_coeff_k_plus_1"] == [1, 1]
        assert result["generating_method_for_ff_coeff_k_plus_n"] == [0, 0]

    def test_readout_indices(self) -> None:
        """Test readout index extraction."""
        mock_repr = _create_mock_machinery_repr(n_readout_indices=[0, 1])
        result = machinery_repr_to_json(mock_repr)

        assert result["readout_macronode_indices"] == [0, 1]

    def test_raise_on_feedforward_in_homodyne_angle(self) -> None:
        """FeedForward homodyne angle is not supported by OptQCClient."""
        mock_repr = _create_mock_machinery_repr()
        ff = FeedForward(_DummyVariable())
        angle_mock = MagicMock()
        angle_mock.theta_a = ff
        angle_mock.theta_b = 2.0
        angle_mock.theta_c = 3.0
        angle_mock.theta_d = 4.0
        mock_repr.get_homodyne_angle.return_value = angle_mock

        with pytest.raises(OptQCClientError, match="does not support FeedForward in MachineOpParam"):
            machinery_repr_to_json(mock_repr)

    def test_raise_on_feedforward_in_displacement(self) -> None:
        """FeedForward displacement is not supported by OptQCClient."""
        mock_repr = _create_mock_machinery_repr()
        ff = FeedForward(_DummyVariable())
        mock_repr.get_displacement_k_minus_1.return_value = (ff, 0.2)

        with pytest.raises(OptQCClientError, match="does not support FeedForward in MachineOpParam"):
            machinery_repr_to_json(mock_repr)


class TestGenerationMethodsFromRealGraph:
    """Test generation method serialization from a real lowered graph."""

    def test_default_policy_round_trip(self) -> None:
        """The default (teleport) policy yields the reference generation methods."""
        graph = _create_reference_graph()
        machinery = MachineryRepr.from_graph_repr(graph)
        result = machinery_repr_to_json(machinery)

        expected = [1, 1, 1, 1, 0, 1]
        assert result["generating_method_for_ff_coeff_k_plus_1"] == expected
        assert result["generating_method_for_ff_coeff_k_plus_n"] == expected
        n_total = graph.n_local_macronodes * graph.n_steps
        assert len(result["generating_method_for_ff_coeff_k_plus_1"]) == n_total
        assert len(result["generating_method_for_ff_coeff_k_plus_n"]) == n_total

    def test_emitted_values_are_plain_ints(self) -> None:
        """The JSON payload emits plain ints rather than enum members."""
        result = machinery_repr_to_json(MachineryRepr.from_graph_repr(_create_reference_graph()))

        for value in (
            *result["generating_method_for_ff_coeff_k_plus_1"],
            *result["generating_method_for_ff_coeff_k_plus_n"],
        ):
            assert type(value) is int
            assert not isinstance(value, FFCoeffMatrixGenerationMethods)

    def test_measurement_macronode_is_zero_filled_by_position(self) -> None:
        """The measurement macronode is ZERO_FILLED and its neighbors are not."""
        graph = _create_reference_graph()
        result = machinery_repr_to_json(MachineryRepr.from_graph_repr(graph))
        measurement_index = graph.get_index(0, 2)

        assert result["generating_method_for_ff_coeff_k_plus_1"][measurement_index] == 0
        assert result["generating_method_for_ff_coeff_k_plus_n"][measurement_index] == 0
        other_values = [
            value
            for index, value in enumerate(result["generating_method_for_ff_coeff_k_plus_1"])
            if index != measurement_index
        ]
        assert other_values == [1] * len(other_values)
        assert result["readout_macronode_indices"] == [measurement_index]

    def test_measure_policy_differs_from_default(self) -> None:
        """The measure policy zero-fills empty wires and differs from the default."""
        graph = _create_reference_graph()
        default_result = machinery_repr_to_json(MachineryRepr.from_graph_repr(graph))
        measure_result = machinery_repr_to_json(MachineryRepr.from_graph_repr(graph, empty_wire_policy="measure"))

        expected = [1, 0, 1, 0, 0, 0]
        assert measure_result["generating_method_for_ff_coeff_k_plus_1"] == expected
        assert measure_result["generating_method_for_ff_coeff_k_plus_n"] == expected
        assert (
            measure_result["generating_method_for_ff_coeff_k_plus_1"]
            != default_result["generating_method_for_ff_coeff_k_plus_1"]
        )

    def test_without_measurement_all_from_macronode_angle(self) -> None:
        """A graph without measurement emits FROM_MACRONODE_ANGLE everywhere."""
        graph = GraphRepr(2, 3)
        graph.place_operation(Initialization((0, 0), 0.0, (BLANK_MODE, 0)))
        graph.place_operation(PhaseRotation((0, 1), pi / 4, swap=False))
        result = machinery_repr_to_json(MachineryRepr.from_graph_repr(graph))

        n_total = graph.n_local_macronodes * graph.n_steps
        assert result["generating_method_for_ff_coeff_k_plus_1"] == [1] * n_total
        assert result["generating_method_for_ff_coeff_k_plus_n"] == [1] * n_total


class TestJsonToMachineryResult:
    """Test json_to_machinery_result conversion."""

    def test_single_shot_single_readout(self) -> None:
        """Test single shot result decoding."""
        data = {
            "readout_macronode_indices": [0],
            "shots": [
                {"values": [0.1, 0.2, 0.3, 0.4]},
            ],
        }
        result = json_to_machinery_result(data)

        assert len(result.measured_vals) == 1
        shot = result.measured_vals[0]
        assert len(shot.items) == 1

        mv = shot.items[0]
        assert mv.m_a == pytest.approx(0.1)
        assert mv.m_b == pytest.approx(0.2)
        assert mv.m_c == pytest.approx(0.3)
        assert mv.m_d == pytest.approx(0.4)
        assert mv.index == 0

    def test_multiple_shots(self) -> None:
        """Test multiple shots result decoding."""
        data = {
            "readout_macronode_indices": [0],
            "shots": [
                {"values": [1.0, 2.0, 3.0, 4.0]},
                {"values": [5.0, 6.0, 7.0, 8.0]},
            ],
        }
        result = json_to_machinery_result(data)

        assert len(result.measured_vals) == 2

    def test_multiple_readout_macronodes(self) -> None:
        """Test result decoding with multiple readout indices."""
        data = {
            "readout_macronode_indices": [0, 1],
            "shots": [
                {"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]},
            ],
        }
        result = json_to_machinery_result(data)

        shot = result.measured_vals[0]
        assert len(shot.items) == 2
        assert shot.items[0].index == 0
        assert shot.items[1].index == 1
