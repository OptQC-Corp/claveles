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

"""Tests for laying two circuits side by side with ``&``."""

from __future__ import annotations

from math import pi

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import BosonicState, QuantumState
from claveles.feedforward import feedforward


def _squeezed_on(mode: int, name: str = "circuit") -> CircuitRepr:
    """Build a one-operation circuit that places a squeezed state on a mode."""
    circuit = CircuitRepr(name)
    circuit.Q(mode) | QuantumState.squeezed(r=0.5)
    return circuit


def test_the_second_circuit_is_laid_above_the_first() -> None:
    left = _squeezed_on(0, "left")
    right = _squeezed_on(0, "right")

    both = left & right

    assert both.name == "left & right"
    assert both.modes == [0, 1]
    assert both.n_operations == 2


def test_the_left_modes_keep_their_numbers_and_their_gaps() -> None:
    """A circuit's mode numbers are how its results are read back, so they are not renumbered."""
    left = CircuitRepr("left")
    left.Q(0) | QuantumState.vacuum()
    left.Q(2) | QuantumState.vacuum()
    right = _squeezed_on(0, "right")

    both = left & right

    # Mode 1 was never part of `left` and stays out of the result; `right` starts above mode 2.
    assert both.modes == [0, 2, 3]


def test_a_mode_named_without_an_operation_is_moved_too() -> None:
    """Naming a mode brings it into the circuit, so it has to be moved out of the way as well."""
    left = CircuitRepr("left")
    left.Q(5)
    right = _squeezed_on(0, "right")

    both = left & right

    assert both.modes == [5, 6]


def test_an_empty_circuit_is_the_identity() -> None:
    circuit = _squeezed_on(0)
    circuit.Q(0) | std.PhaseRotation(pi / 2)
    empty = CircuitRepr("empty")

    for both in (circuit & empty, empty & circuit):
        assert both.modes == circuit.modes
        assert both.n_operations == circuit.n_operations
        assert str(both) == str(circuit)


def test_the_operands_are_left_untouched() -> None:
    left = _squeezed_on(0, "left")
    right = _squeezed_on(0, "right")

    _ = left & right

    assert left.modes == [0]
    assert right.modes == [0]
    assert left.n_operations == right.n_operations == 1


def test_the_operations_are_copied_rather_than_shared() -> None:
    """An operation instance may only be applied once, so the result cannot reuse the operands'."""
    left = _squeezed_on(0, "left")
    right = _squeezed_on(0, "right")

    both = left & right

    assert both.get_operation(0) is not left.get_operation(0)
    assert both.get_operation(1) is not right.get_operation(0)
    # The operands go on owning their own operations.
    assert left.placements[0].owner is left
    assert both.placements[0].owner is both
    assert both.placements[1].modes == (1,)


def test_arity_and_open_modes_add_up() -> None:
    left = _squeezed_on(0, "left")
    left.Q(0) | std.MeasureHomodyne(0.0)
    right = CircuitRepr("right")
    right.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)

    both = left & right

    assert both.open_modes == [1, 2]
    assert both.input_arity == 2
    # Mode 0 was measured away in `left`; both of `right`'s modes survive.
    assert both.output_arity == 2


def test_placed_components_move_with_their_modes() -> None:
    left = _squeezed_on(0, "left")
    right = CircuitRepr("right")
    right.Q(0, 1) | QuantumState.epr(r=0.5)

    both = left & right

    modes = [modes for modes, _ in both.placed_components]
    assert modes == [[0], [1, 2]]


def test_feedforward_within_a_circuit_survives_composition() -> None:
    @feedforward
    def double(x: float) -> float:
        return x * 2.0

    right = CircuitRepr("right")
    right.Q(0) | QuantumState.vacuum()
    right.Q(1) | QuantumState.vacuum()
    measured = right.Q(0) | std.MeasureHomodyne(0.0)
    right.Q(1) | std.PhaseRotation(double(measured))

    both = _squeezed_on(0, "left") & right

    proto = both.proto()
    assert len(proto.nlffs) == 1
    # The feedforward reads the measurement, which is the first operation of `right` to be serialized.
    # State placements travel separately, so the measurement is at index 0 of the operation list.
    assert proto.nlffs[0].from_operation == 0
    assert proto.nlffs[0].to_operation == 1


def test_a_measured_variable_goes_on_referring_to_its_own_circuit() -> None:
    """The result is an independent circuit, so a variable taken from an operand cannot reach it."""
    left = _squeezed_on(0, "left")
    measured = left.Q(0) | std.MeasureHomodyne(0.0)

    both = left & _squeezed_on(0, "right")

    assert measured.get_from_operation() is left.get_operation(1)
    assert measured.get_from_operation() is not both.get_operation(1)


def test_composition_is_associative() -> None:
    first = _squeezed_on(0, "first")
    second = CircuitRepr("second")
    second.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    third = _squeezed_on(1, "third")

    left_first = (first & second) & third
    right_first = first & (second & third)

    assert left_first.modes == right_first.modes
    assert str(left_first) == str(right_first)


def test_a_circuit_can_be_composed_with_itself() -> None:
    circuit = _squeezed_on(0)
    circuit.Q(0) | std.PhaseRotation(pi / 2)

    both = circuit & circuit

    assert both.modes == [0, 1]
    assert both.n_operations == 4
    assert both.get_operation(0) is not both.get_operation(2)


def test_the_result_serializes() -> None:
    left = _squeezed_on(0, "left")
    left.Q(0) | std.MeasureHomodyne(0.0)
    right = _squeezed_on(0, "right")

    proto = (left & right).proto()

    assert proto.n_modes == 2
    assert len(proto.initial_states) == 2
    assert len(proto.operations) == 1


def test_an_open_mode_of_either_circuit_still_needs_a_default() -> None:
    left = CircuitRepr("left")
    left.Q(0) | std.PhaseRotation(pi / 2)

    both = left & _squeezed_on(0, "right")

    with pytest.raises(ValueError, match="no state placed"):
        both.proto()
    assert both.proto(default_state=BosonicState.vacuum()).n_modes == 2


def test_composing_with_something_that_is_not_a_circuit_is_rejected() -> None:
    with pytest.raises(TypeError):
        _ = _squeezed_on(0) & 1  # ty: ignore[unsupported-operator]
