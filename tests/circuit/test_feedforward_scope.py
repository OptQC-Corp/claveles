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

"""Tests for which measurements a feedforward may read: order, reachability and composition."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import QuantumState
from claveles.feedforward import feedforward

if TYPE_CHECKING:
    from collections.abc import Iterator


@feedforward
def double(x: float) -> float:
    return x * 2.0


def _measure_then_rotate() -> tuple[CircuitRepr, std.PhaseRotation]:
    """Build a circuit that measures mode 0 and rotates mode 1 from the result."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)
    rotation = std.PhaseRotation(double(measured))
    circuit.Q(1) | rotation
    return circuit, rotation


def _measuring_and_reading_halves() -> tuple[CircuitRepr, CircuitRepr]:
    """Build one circuit that measures and another whose gate reads that measurement."""
    measuring = CircuitRepr("measuring")
    measuring.Q(0) | QuantumState.vacuum()
    measured = measuring.Q(0) | std.MeasureHomodyne(0.0)
    reading = CircuitRepr("reading")
    reading.Q(0) | QuantumState.vacuum()
    reading.Q(0) | std.PhaseRotation(double(measured))
    return measuring, reading


def test_a_feedforward_may_read_a_measurement_placed_before_it() -> None:
    circuit, _ = _measure_then_rotate()

    nlffs = circuit.proto().nlffs

    assert [(ff.from_operation, ff.to_operation) for ff in nlffs] == [(0, 1)]


def test_editing_a_feedforward_in_front_of_its_measurement_is_rejected() -> None:
    """Moving the reading operation before the measurement breaks a circuit that was fine."""
    circuit, rotation = _measure_then_rotate()
    circuit.remove(3)

    with pytest.raises(RuntimeError, match="comes after it"):
        circuit.append(rotation, idx=2, mode=1)


def test_a_rejected_edit_leaves_the_circuit_serializable() -> None:
    circuit, rotation = _measure_then_rotate()
    circuit.remove(3)
    with pytest.raises(RuntimeError, match="comes after it"):
        circuit.append(rotation, idx=2, mode=1)

    assert circuit.n_operations == 3
    assert circuit.proto().nlffs == []


def test_replacing_an_operation_with_one_that_reads_a_later_measurement_is_rejected() -> None:
    """Changing a parameter is where a feedforward can be put in front of its measurement."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()
    circuit.Q(1) | std.Squeezing(1.0)
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)

    with pytest.raises(RuntimeError, match="comes after it"):
        circuit.replace_operation(2, std.Squeezing(double(measured)))


def test_an_operation_cannot_be_given_a_feedforward_by_writing_to_it() -> None:
    """An attribute write goes through no placement, so it is the one route no check would see."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()
    squeezing = std.Squeezing(1.0)
    circuit.Q(1) | squeezing
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)

    with pytest.raises(AttributeError, match="immutable once built"):
        squeezing.r = double(measured)

    assert circuit.proto().nlffs == []


def test_serializing_checks_the_order_as_well_as_editing_does() -> None:
    """The last gate before a circuit leaves the SDK, however the sequence came to be broken."""
    circuit, _ = _measure_then_rotate()
    # Reached through the private sequence on purpose: it stands in for an internal path that edits the
    # circuit without rechecking it, which is exactly what this check is here to catch.
    placements = circuit._placements  # noqa: SLF001
    placements[2], placements[3] = placements[3], placements[2]

    with pytest.raises(RuntimeError, match="comes after it"):
        circuit.proto()


def test_reading_a_measurement_this_circuit_dropped_is_rejected_when_placed() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)
    circuit.remove(2)

    with pytest.raises(RuntimeError, match="was removed from this circuit"):
        circuit.Q(1) | std.PhaseRotation(double(measured))


def test_reading_another_circuit_s_measurement_is_allowed_while_building() -> None:
    """Composing the two resolves it, so the half-built state is not an error in itself."""
    _, reading = _measuring_and_reading_halves()

    assert reading.n_operations == 2


def test_composing_resolves_a_feedforward_that_reads_across_the_two_circuits() -> None:
    measuring, reading = _measuring_and_reading_halves()

    both = measuring & reading

    nlffs = both.proto().nlffs
    assert [(ff.from_operation, ff.to_operation) for ff in nlffs] == [(0, 1)]


def test_composing_in_the_order_that_puts_the_measurement_last_is_rejected() -> None:
    measuring, reading = _measuring_and_reading_halves()

    with pytest.raises(RuntimeError, match="comes after it"):
        _ = reading & measuring


def test_a_circuit_composed_with_itself_keeps_its_own_feedforward() -> None:
    """The two halves are independent copies, so neither reads the other's measurement."""
    circuit, _ = _measure_then_rotate()

    both = circuit & circuit

    assert both.n_operations == 8
    nlffs = [(ff.from_operation, ff.to_operation) for ff in both.proto().nlffs]
    assert sorted(nlffs) == [(0, 1), (2, 3)]


def test_a_feedforward_reading_into_an_earlier_composition_is_reported() -> None:
    measuring, reading = _measuring_and_reading_halves()
    middle = CircuitRepr("middle")
    middle.Q(0) | QuantumState.vacuum()

    nested = (measuring & middle) & reading

    with pytest.raises(ValueError, match="from the circuit 'measuring'"):
        nested.proto()


def test_adding_a_feedforward_to_a_composed_circuit_is_reported() -> None:
    measuring = CircuitRepr("measuring")
    measuring.Q(0) | QuantumState.vacuum()
    measured = measuring.Q(0) | std.MeasureHomodyne(0.0)
    other = CircuitRepr("other")
    other.Q(0) | QuantumState.vacuum()

    composed = measuring & other
    composed.Q(1) | std.PhaseRotation(double(measured))

    with pytest.raises(ValueError, match="does not reach the composed circuit"):
        composed.proto()


def test_the_message_names_the_circuit_and_the_position() -> None:
    _, reading = _measuring_and_reading_halves()

    with pytest.raises(ValueError, match=r"from the circuit 'measuring' at position 1"):
        reading.proto()


def test_appending_an_operation_that_reads_a_dropped_measurement_is_rejected() -> None:
    circuit, rotation = _measure_then_rotate()
    circuit.remove([2, 3])

    with pytest.raises(RuntimeError, match="was removed from this circuit"):
        circuit.append(rotation, idx=2, mode=1)

    assert circuit.n_operations == 2
    assert circuit.modes == [0, 1]


def test_lowering_keeps_a_feedforward_readable() -> None:
    circuit, _ = _measure_then_rotate()

    circuit.convert_std_ops_to_intrinsic()

    assert [(ff.from_operation, ff.to_operation) for ff in circuit.proto().nlffs] == [(0, 1)]


def test_lowering_a_composed_circuit_keeps_a_crossing_feedforward() -> None:
    measuring, reading = _measuring_and_reading_halves()
    both = measuring & reading

    both.convert_std_ops_to_intrinsic()

    assert [(ff.from_operation, ff.to_operation) for ff in both.proto().nlffs] == [(0, 1)]


def test_a_gate_with_a_plain_parameter_is_unaffected() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.PhaseRotation(pi / 2)

    assert circuit.proto().nlffs == []


class _CountingPlacements(list):  # noqa: FURB189  # a real list, so it still fits the attribute
    """A placement list that records how often the whole sequence is walked.

    Only iteration is counted, which is what a check that consults the earlier placements does. An
    exact count is asserted, so a scan reaching the sequence by some other route shows up as a
    mismatch rather than passing quietly.
    """

    def __init__(self, placements: list) -> None:
        super().__init__(placements)
        self.scans = 0

    def __iter__(self) -> Iterator:
        self.scans += 1
        return super().__iter__()


def _count_scans(circuit: CircuitRepr) -> _CountingPlacements:
    """Make the circuit count the times its placements are walked from here on."""
    counting = _CountingPlacements(circuit._placements)  # noqa: SLF001  # no public seam for this
    circuit._placements = counting  # noqa: SLF001
    return counting


def test_placing_an_operation_without_feedforward_does_not_walk_the_sequence() -> None:
    """Building a circuit stays linear only while insertions do not scan what came before."""
    circuit = CircuitRepr("test")
    counting = _count_scans(circuit)

    for mode in range(50):
        circuit.Q(mode) | QuantumState.vacuum()
        circuit.Q(mode) | std.PhaseRotation(pi / 2)

    assert circuit.n_operations == 100
    assert counting.scans == 0


def test_placing_an_operation_with_feedforward_walks_the_sequence_once() -> None:
    """The scan the operation above skips is the one that finds the measurement being read."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)
    counting = _count_scans(circuit)

    circuit.Q(1) | std.PhaseRotation(double(measured))

    assert counting.scans == 1
