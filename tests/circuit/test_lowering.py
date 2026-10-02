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

"""Tests for lowering a circuit's operations to their intrinsic form in place."""

from __future__ import annotations

from copy import deepcopy
from math import pi

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic, std
from claveles.circuit.state import BosonicState, QuantumState
from claveles.feedforward import feedforward


@feedforward
def double(x: float) -> float:
    return x * 2.0


def test_a_circuit_with_a_placed_state_can_be_lowered() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.squeezed(r=0.5)
    circuit.Q(0) | std.PhaseRotation(pi / 2)

    circuit.convert_std_ops_to_intrinsic()

    assert circuit.get_operation(0).is_state_preparation()
    assert isinstance(circuit.get_operation(1), intrinsic.PhaseRotation)
    modes = [modes for modes, _ in circuit.placed_components]
    assert modes == [[0]]


def test_a_circuit_with_a_measurement_can_be_lowered() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()
    _ = circuit.Q(0) | std.MeasureHomodyne(0.0)

    before = (circuit.open_modes, circuit.input_arity, circuit.output_arity)
    circuit.convert_std_ops_to_intrinsic()

    assert (circuit.open_modes, circuit.input_arity, circuit.output_arity) == before
    assert circuit.get_operation(2).is_measurement_op()


def test_a_lowered_operation_lands_on_the_modes_it_acts_on() -> None:
    """A beam splitter's phase rotations take a single mode each, not both of the gate's modes."""
    circuit = CircuitRepr("test")
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)

    circuit.convert_std_ops_to_intrinsic()

    assert [placement.modes for placement in circuit.placements] == [(0,), (1,), (0, 1), (0,), (1,)]


def test_a_decomposition_reports_the_modes_each_part_acts_on() -> None:
    """The parts come back with modes taken from the ones passed in.

    Asked of an operation that was never placed, so the decomposition is shown to depend on the modes
    it is given rather than on where the operation sits. The modes are not 0 and 1, which an
    implementation that ignored the argument would still get right.
    """
    lowered = std.BeamSplitter(theta=pi / 4, phi=0.0).to_intrinsic_ops([3, 7])

    assert [modes for _, modes in lowered] == [[3], [7], [3, 7], [3], [7]]


def test_serializing_an_operation_uses_the_modes_it_is_given() -> None:
    (op,) = intrinsic.PhaseRotation(0.5).proto([4])

    assert list(op.modes) == [4]


def test_a_lowered_operation_belongs_to_the_circuit_it_was_lowered_in() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.Squeezing(r=1.0)

    circuit.convert_std_ops_to_intrinsic()

    for placement in circuit.placements:
        assert placement.owner is circuit


def test_copying_a_placement_does_not_copy_the_circuit_it_belongs_to() -> None:
    """A placement records its circuit, so a copy of one has to be kept from dragging the circuit along.

    Composition copies placements, and a copy that pulled a whole circuit with it would leave the
    feedforward inside pointing at that copy rather than at the circuit being built.
    """
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.Squeezing(r=1.0)
    (placement,) = circuit.placements[1:]

    copied = deepcopy(placement, {id(circuit): circuit})

    assert copied is not placement
    assert copied.op is not placement.op
    assert copied.owner is circuit


def test_feedforward_survives_lowering() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)
    circuit.Q(1) | std.PhaseRotation(double(measured))

    before = [(ff.from_operation, ff.to_operation) for ff in circuit.proto().nlffs]
    circuit.convert_std_ops_to_intrinsic()

    assert [(ff.from_operation, ff.to_operation) for ff in circuit.proto().nlffs] == before


def test_a_measured_variable_follows_the_circuit_through_lowering() -> None:
    """The variable is what a caller holds on to, so it has to point into the lowered sequence."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)
    circuit.Q(1) | std.PhaseRotation(double(measured))

    circuit.convert_std_ops_to_intrinsic()

    assert measured.get_from_placement() is circuit._placements[2]  # noqa: SLF001  # no public accessor
    assert measured.get_from_operation() is circuit.get_operation(2)


def test_lowering_is_idempotent() -> None:
    circuit = CircuitRepr("test")
    for mode in (0, 1, 2):
        circuit.Q(mode) | QuantumState.vacuum()
    measured = circuit.Q(2) | std.MeasureHomodyne(0.0)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    circuit.Q(1) | std.PhaseRotation(double(measured))

    circuit.convert_std_ops_to_intrinsic()
    once = str(circuit)
    once_nlffs = [(ff.from_operation, ff.to_operation) for ff in circuit.proto().nlffs]
    circuit.convert_std_ops_to_intrinsic()

    assert str(circuit) == once
    assert [(ff.from_operation, ff.to_operation) for ff in circuit.proto().nlffs] == once_nlffs


def test_lowering_agrees_with_serializing_the_decomposition() -> None:
    """The destructive path and `proto(decompose_std=True)` have to give the same expansion."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    circuit.Q(0) | std.Squeezing(r=1.0)

    decomposed_on_the_fly = circuit.proto(decompose_std=True)
    circuit.convert_std_ops_to_intrinsic()

    assert circuit.proto() == decomposed_on_the_fly


def test_a_composed_circuit_can_be_lowered() -> None:
    left = CircuitRepr("left")
    left.Q(0) | QuantumState.vacuum()
    measured = left.Q(0) | std.MeasureHomodyne(0.0)
    left.Q(1) | std.PhaseRotation(double(measured))
    right = CircuitRepr("right")
    right.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)

    both = left & right
    both.convert_std_ops_to_intrinsic()

    proto = both.proto(default_state=BosonicState.vacuum())
    assert len(proto.nlffs) == 1
    assert proto.nlffs[0].from_operation < proto.nlffs[0].to_operation
    assert all(placement.owner is both for placement in both.placements)
