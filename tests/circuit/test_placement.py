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

"""Tests for the circuit as a sequence of placements: validation, arity and editing."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np
import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import (
    BosonicState,
    FockState,
    GaussianState,
    HardwareConstrainedSqueezedState,
    QuantumState,
)
from claveles.feedforward import feedforward

if TYPE_CHECKING:
    from collections.abc import Callable

    from claveles.circuit.state import InitialState


def test_a_state_placement_is_an_operation_in_the_sequence() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.PhaseRotation(pi / 2)

    assert circuit.n_operations == 2
    assert circuit.get_operation(0).is_state_preparation()
    assert not circuit.get_operation(1).is_state_preparation()


def test_placing_a_state_twice_on_a_mode_is_rejected() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()

    with pytest.raises(RuntimeError, match="already been used"):
        circuit.Q(0) | QuantumState.vacuum()


def test_placing_a_state_after_a_gate_is_rejected() -> None:
    """The first operation to touch a mode settles where its input comes from."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)

    # Mode 1 was settled as an open input by the beam splitter.
    with pytest.raises(RuntimeError, match="already been used"):
        circuit.Q(1) | QuantumState.vacuum()


def test_operating_on_a_removed_mode_is_rejected() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.MeasureHomodyne(0.0)

    with pytest.raises(RuntimeError, match="already measured or traced out"):
        circuit.Q(0) | std.PhaseRotation(pi / 2)


def test_partial_trace_consumes_its_operands() -> None:
    circuit = CircuitRepr("test")
    for mode in (0, 1):
        circuit.Q(mode) | QuantumState.vacuum()
    circuit.Q(1) | std.PartialTrace()

    assert circuit.output_arity == 1
    with pytest.raises(RuntimeError, match="already measured or traced out"):
        circuit.Q(1) | std.PhaseRotation(0.0)


def test_partial_trace_with_keep_consumes_every_other_mode() -> None:
    circuit = CircuitRepr("test")
    for mode in (0, 1, 2):
        circuit.Q(mode) | QuantumState.vacuum()
    circuit.Q(1) | std.PartialTrace(keep=True)

    assert circuit.output_arity == 1
    with pytest.raises(RuntimeError, match="already measured or traced out"):
        circuit.Q(0) | std.PhaseRotation(0.0)


def test_arity_of_an_open_circuit() -> None:
    """The first example of note_5058673: a placement then a two-mode gate gives 1 -> 2."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.squeezed(r=0.5)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)

    assert circuit.open_modes == [1]
    assert circuit.input_arity == 1
    assert circuit.output_arity == 2


def test_arity_with_an_open_mode_that_is_measured() -> None:
    """The second example of note_5058673: a placement plus a measurement elsewhere gives 1 -> 1."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.squeezed(r=0.5)
    circuit.Q(1) | std.MeasureHomodyne(0.0)

    assert circuit.open_modes == [1]
    assert circuit.input_arity == 1
    assert circuit.output_arity == 1


def test_arity_of_a_closed_circuit_is_zero_in() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.PhaseRotation(0.0)

    assert circuit.input_arity == 0
    assert circuit.output_arity == 1


def test_remove_drops_an_operation_and_renumbers() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.PhaseRotation(pi / 2)
    circuit.Q(0) | std.Squeezing(1.0)

    circuit.remove(1)

    assert circuit.n_operations == 2
    assert isinstance(circuit.get_operation(1), std.Squeezing)


def test_remove_accepts_several_positions() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.PhaseRotation(pi / 2)
    circuit.Q(0) | std.Squeezing(1.0)

    circuit.remove([1, 2])

    assert circuit.n_operations == 1


def test_remove_rejects_a_position_that_does_not_exist() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()

    with pytest.raises(IndexError, match="No operation at position 5"):
        circuit.remove(5)


def test_remove_revalidates_the_rest() -> None:
    """Dropping a placement can leave a later placement on a mode that is no longer untouched."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.0)
    circuit.Q(2) | QuantumState.vacuum()

    # Removing the beam splitter is fine: nothing depends on it.
    circuit.remove(1)
    assert circuit.n_operations == 2


def test_remove_leaves_the_circuit_untouched_when_it_fails() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.MeasureHomodyne(0.0)

    with pytest.raises(IndexError):
        circuit.remove(9)

    assert circuit.n_operations == 2
    assert circuit.output_arity == 0


def test_remove_refuses_to_orphan_a_feedforward() -> None:
    @feedforward
    def double(x: float) -> float:
        return 2.0 * x

    circuit = CircuitRepr("test")
    for mode in (0, 1):
        circuit.Q(mode) | QuantumState.vacuum()
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)
    circuit.Q(1) | std.PhaseRotation(double(measured))

    with pytest.raises(RuntimeError, match="read by a feedforward"):
        circuit.remove(2)


def test_append_inserts_at_a_position() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.Squeezing(1.0)

    circuit.append(std.PhaseRotation(pi / 2), idx=1, mode=0)

    assert circuit.n_operations == 3
    assert isinstance(circuit.get_operation(1), std.PhaseRotation)
    assert isinstance(circuit.get_operation(2), std.Squeezing)


def test_append_closes_an_open_circuit() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | std.PhaseRotation(pi / 2)
    assert circuit.input_arity == 1

    circuit.append(QuantumState.vacuum(), idx=0, mode=0)

    assert circuit.input_arity == 0
    assert circuit.get_operation(0).is_state_preparation()


def test_append_at_a_position_that_breaks_the_sequence_is_rejected() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.PhaseRotation(pi / 2)

    # Placing a second state on mode 0, wherever it goes, conflicts with the first.
    with pytest.raises(RuntimeError, match="already been used"):
        circuit.append(QuantumState.vacuum(), idx=2, mode=0)
    assert circuit.n_operations == 2


def test_append_rejects_the_wrong_number_of_modes() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()

    with pytest.raises(RuntimeError, match="number of modes"):
        circuit.append(std.PhaseRotation(0.0), idx=1, mode=[0, 1])


def test_append_beyond_the_end_is_rejected() -> None:
    circuit = CircuitRepr("test")

    with pytest.raises(IndexError, match="Cannot insert at position 3"):
        circuit.append(std.PhaseRotation(0.0), idx=3, mode=0)


def _measured_then_rotated() -> CircuitRepr:
    """Build a circuit measuring mode 0 and rotating mode 1 by the result."""

    @feedforward
    def double(x: float) -> float:
        return 2.0 * x

    circuit = CircuitRepr("test")
    for mode in (0, 1):
        circuit.Q(mode) | QuantumState.vacuum()
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)
    circuit.Q(1) | std.PhaseRotation(double(measured))
    return circuit


def test_replace_operation_puts_a_new_parameter_in_place() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.Squeezing(1.0)

    circuit.replace_operation(1, circuit.get_operation(1).replace(r=1.5))

    assert circuit.n_operations == 2
    assert circuit.get_operation(1).parameters() == (1.5,)


def test_replace_operation_keeps_the_modes_the_position_names() -> None:
    circuit = CircuitRepr("test")
    for mode in (0, 1):
        circuit.Q(mode) | QuantumState.vacuum()
    circuit.Q(0, 1) | std.BeamSplitter(pi / 4, 0.0)

    circuit.replace_operation(2, std.BeamSplitter(pi / 8, 0.0))

    assert circuit.placements[2].modes == (0, 1)


def test_replace_operation_rejects_an_operation_of_a_different_arity() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.Squeezing(1.0)

    with pytest.raises(RuntimeError, match="number of modes"):
        circuit.replace_operation(1, std.BeamSplitter(pi / 4, 0.0))
    assert isinstance(circuit.get_operation(1), std.Squeezing)


def test_replace_operation_rejects_a_position_that_does_not_exist() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()

    with pytest.raises(IndexError, match="No operation at position 3"):
        circuit.replace_operation(3, std.Squeezing(1.0))


def test_replace_operation_revalidates_the_rest() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.Squeezing(1.0)

    # Turning the squeezing into a second state placement conflicts with the first.
    with pytest.raises(RuntimeError, match="already been used"):
        circuit.replace_operation(1, QuantumState.vacuum())
    assert isinstance(circuit.get_operation(1), std.Squeezing)


def test_replace_operation_carries_a_feedforward_over_to_the_new_measurement() -> None:
    circuit = _measured_then_rotated()

    circuit.replace_operation(2, std.MeasureHomodyne(pi / 2))

    assert circuit.get_operation(2).parameters() == (pi / 2,)
    assert [(ff.from_operation, ff.to_operation) for ff in circuit.proto().nlffs] == [(0, 1)]


def test_replace_operation_refuses_to_orphan_a_feedforward() -> None:
    circuit = _measured_then_rotated()

    with pytest.raises(RuntimeError, match="read by a feedforward"):
        circuit.replace_operation(2, std.Squeezing(1.0))
    assert isinstance(circuit.get_operation(2), std.MeasureHomodyne)


def test_replace_operation_leaves_the_feedforward_alone_when_it_fails() -> None:
    circuit = _measured_then_rotated()
    with pytest.raises(RuntimeError, match="read by a feedforward"):
        circuit.replace_operation(2, std.Squeezing(1.0))

    assert [(ff.from_operation, ff.to_operation) for ff in circuit.proto().nlffs] == [(0, 1)]


def test_replace_operation_accepts_a_state() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.Squeezing(1.0)

    circuit.replace_operation(0, QuantumState.squeezed(r=0.5))

    assert circuit.get_operation(0).is_state_preparation()
    assert circuit.placed_states[0].n_modes == 1  # ty: ignore[unresolved-attribute]


def test_serializing_an_open_circuit_needs_a_default() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | std.PhaseRotation(pi / 2)

    with pytest.raises(ValueError, match=r"Mode\(s\) \[0\] have no state placed"):
        circuit.proto()

    filled = circuit.proto(default_state=HardwareConstrainedSqueezedState())
    assert len(filled.initial_states) == 1
    assert filled.initial_states[0].HasField("squeezed")


def test_an_open_circuit_cannot_be_saved(tmp_path) -> None:  # noqa: ANN001
    circuit = CircuitRepr("test")
    circuit.Q(0) | std.PhaseRotation(pi / 2)

    with pytest.raises(ValueError, match="have no state placed"):
        circuit.save(tmp_path / "circuit.txt")


def test_a_multi_mode_state_covers_all_its_modes() -> None:
    """A state correlating two modes is placed as one state, not as two marginals."""
    circuit = CircuitRepr("test")
    circuit.Q(0, 1) | QuantumState.epr(r=0.5)

    modes, component = circuit.placed_components[0]
    assert modes == [0, 1]
    assert component.n_modes == 2
    # Reading the per-mode view shows the same component covering both modes.
    assert circuit.placed_states[0] is component
    assert circuit.placed_states[1] is component
    assert circuit.input_arity == 0
    assert circuit.output_arity == 2


def test_a_multi_mode_state_names_its_modes_on_the_wire() -> None:
    """The positional form cannot say that one entry spans two modes, so the modes are named."""
    circuit = CircuitRepr("test")
    circuit.Q(0, 1) | QuantumState.epr(r=0.5)

    entries = circuit.proto().initial_states
    assert len(entries) == 1
    assert list(entries[0].modes) == [0, 1]
    # Four quadratures: the x of both modes followed by the p of both.
    assert len(entries[0].bosonic.gaussian_states[0].mean) == 4


def test_a_single_mode_circuit_still_leaves_its_modes_implicit() -> None:
    """What was expressible positionally keeps serializing that way, so old bytes stay old bytes."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()

    for entry in circuit.proto().initial_states:
        assert list(entry.modes) == []


def test_a_joint_fock_state_can_be_placed() -> None:
    """A Fock state correlating two modes goes in as one state, not as two marginals."""
    sv = np.zeros((2, 3), dtype=np.complex128)
    sv[0, 1] = sv[1, 0] = np.sqrt(0.5)
    joint = QuantumState([FockState(cutoff_dims=[2, 3], state_vector=sv)], [[0, 1]])

    circuit = CircuitRepr("test")
    circuit.Q(0, 1) | joint

    modes, component = circuit.placed_components[0]
    assert modes == [0, 1]
    assert component.n_modes == 2
    assert list(circuit.proto().initial_states[0].modes) == [0, 1]
    assert circuit.proto().initial_states[0].HasField("fock_multi_mode")


def test_a_wider_joint_fock_state_is_refused() -> None:
    """Preparing a joint Fock state goes through the gate path, which reaches three sites at most."""
    pure = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    pure[0, 0, 0, 0] = 1.0
    four_mode = QuantumState([FockState(cutoff_dims=[2, 2, 2, 2], state_vector=pure)], [[0, 1, 2, 3]])

    circuit = CircuitRepr("test")
    with pytest.raises(RuntimeError, match="at most 3 mode"):
        circuit.Q(0, 1, 2, 3) | four_mode

    # Purifying a mixed state spends one of those three sites on the ancilla.
    dm = np.zeros((8, 8), dtype=np.complex128)
    dm[0, 0] = 1.0
    three_mode_mixed = QuantumState([FockState(cutoff_dims=[2, 2, 2], density_matrix=dm)], [[0, 1, 2]])

    other = CircuitRepr("test")
    with pytest.raises(RuntimeError, match="at most 2 mode"):
        other.Q(0, 1, 2) | three_mode_mixed


def test_separable_components_are_placed_one_per_mode() -> None:
    """A container of single-mode components spans several modes without needing multi-mode support."""
    circuit = CircuitRepr("test")
    circuit.Q(0, 1) | (QuantumState.vacuum() & QuantumState.squeezed(r=0.5))

    states = circuit.placed_states
    assert sorted(states) == [0, 1]
    assert all(isinstance(state, BosonicState) for state in states.values())
    assert circuit.input_arity == 0
    assert circuit.output_arity == 2


def test_a_circuit_uses_only_the_modes_it_names() -> None:
    """Naming a mode says nothing about the modes below it."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(9) | QuantumState.squeezed(r=0.5)

    assert circuit.modes == [0, 9]
    # The count is of the modes in use, not of the range the indices span.
    assert circuit.n_modes == 2
    assert circuit.open_modes == []
    assert circuit.output_arity == 2


def test_a_gap_is_not_an_open_mode() -> None:
    """A mode the circuit never names is outside the circuit, not an input it expects."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(2) | QuantumState.vacuum()

    assert circuit.modes == [0, 2]
    assert circuit.open_modes == []
    assert circuit.input_arity == 0
    assert sorted(circuit.placed_states) == [0, 2]


def test_a_circuit_with_a_gap_serializes() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(2) | QuantumState.squeezed(r=0.5)
    circuit.Q(0, 2) | std.BeamSplitter(theta=0.3, phi=0.0)

    proto = circuit.proto()

    assert proto.n_modes == 2
    # The modes cannot be read off the positions once they skip a value, so they are named.
    assert [list(entry.modes) for entry in proto.initial_states] == [[0], [2]]


def test_a_contiguous_circuit_still_serializes_positionally() -> None:
    """What the wire carries for an ordinary circuit is unchanged, which the golden digests rely on."""
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(1) | QuantumState.vacuum()

    proto = circuit.proto()

    assert [list(entry.modes) for entry in proto.initial_states] == [[], []]


def test_placed_states_is_keyed_by_mode() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(3) | QuantumState.vacuum()
    circuit.Q(5) | std.PhaseRotation(0.0)

    states = circuit.placed_states
    assert sorted(states) == [3, 5]
    assert states[3] is not None
    # Mode 5 was acted on without a state being placed, so it is an open input.
    assert states[5] is None
    assert circuit.open_modes == [5]


def test_a_partial_trace_that_keeps_a_mode_consumes_only_the_modes_in_use() -> None:
    """`keep` discards every *other* mode of the circuit, which is not every index below the highest."""
    circuit = CircuitRepr("test")
    for mode in (0, 4, 7):
        circuit.Q(mode) | QuantumState.vacuum()
    circuit.Q(4) | std.PartialTrace(keep=True)

    assert circuit.output_arity == 1
    with pytest.raises(RuntimeError, match="already measured or traced out"):
        circuit.Q(7) | std.PhaseRotation(0.0)


def _wide_gaussian() -> BosonicState:
    """Build a two-mode bosonic state, to be refused as a default.

    Returns:
        BosonicState: A two-mode vacuum held as a single peak.
    """
    return BosonicState(
        np.fromiter([1], dtype=np.complex128),
        [GaussianState(np.zeros(4, dtype=np.complex128), 0.5 * np.eye(4, dtype=np.float64))],
    )


def test_a_default_state_may_be_given_as_a_container() -> None:
    def circuit() -> CircuitRepr:
        built = CircuitRepr("test")
        built.Q(0) | std.PhaseRotation(pi / 2)
        return built

    through_container = circuit().proto(default_state=QuantumState.hardware_squeezed())
    directly = circuit().proto(default_state=HardwareConstrainedSqueezedState())

    assert through_container.SerializeToString() == directly.SerializeToString()


def test_a_default_state_of_several_components_is_refused() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | std.PhaseRotation(pi / 2)

    with pytest.raises(ValueError, match="holds 2 components"):
        circuit.proto(default_state=QuantumState.vacuum() & QuantumState.vacuum())


@pytest.mark.parametrize(
    "wrap",
    [lambda state: state, lambda state: QuantumState([state])],
    ids=["basic", "container"],
)
def test_a_default_state_of_several_modes_is_refused(wrap: Callable[[BosonicState], InitialState]) -> None:
    # A mode gap makes the entries non-positional, which is the shape where serializing a wider default
    # without complaint would produce a file that could not be read back.
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.PhaseRotation(pi / 2)
    circuit.Q(5) | std.PhaseRotation(pi / 2)

    with pytest.raises(ValueError, match="spans 2 modes"):
        circuit.proto(default_state=wrap(_wide_gaussian()))


def test_a_default_state_is_checked_even_when_no_mode_is_open() -> None:
    # Nothing would have used it, but a default that could never fill a mode is still not a default.
    circuit = CircuitRepr("test")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.PhaseRotation(pi / 2)

    with pytest.raises(ValueError, match="spans 2 modes"):
        circuit.proto(default_state=_wide_gaussian())
