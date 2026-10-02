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

"""Tests for feeding one circuit's output into another's open inputs with ``@``."""

from __future__ import annotations

import logging
from math import pi

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import BosonicState, QuantumState
from claveles.feedforward import feedforward


@feedforward
def double(x: float) -> float:
    return x * 2.0


def _source(mode: int = 0, name: str = "source") -> CircuitRepr:
    """Build a circuit that places a state on a mode and leaves it carrying one."""
    circuit = CircuitRepr(name)
    circuit.Q(mode) | QuantumState.squeezed(r=0.5)
    return circuit


def _sink(mode: int = 0, name: str = "sink") -> CircuitRepr:
    """Build a circuit that acts on a mode without placing a state, leaving it open."""
    circuit = CircuitRepr(name)
    circuit.Q(mode) | std.PhaseRotation(pi / 2)
    return circuit


def test_a_matching_mode_number_is_a_connection() -> None:
    whole = _sink() @ _source()

    assert whole.name == "sink @ source"
    assert whole.modes == [0]
    assert whole.input_arity == 0
    assert whole.output_arity == 1
    assert whole.n_operations == 2


def test_the_upstream_operations_come_first() -> None:
    """`c2 @ c1` runs c1 first, the way f∘g applies g first."""
    whole = _sink() @ _source()

    assert whole.get_operation(0).is_state_preparation()
    assert isinstance(whole.get_operation(1), std.PhaseRotation)


def test_nothing_is_renumbered() -> None:
    upstream = _source(mode=3, name="upstream")
    downstream = _sink(mode=3, name="downstream")

    whole = downstream @ upstream

    assert whole.modes == [3]
    assert [placement.modes for placement in whole.placements] == [(3,), (3,)]


def test_an_output_the_downstream_circuit_does_not_take_stays_an_output() -> None:
    upstream = _source(name="upstream")
    upstream.Q(3) | QuantumState.vacuum()

    whole = _sink() @ upstream

    assert whole.output_modes == [0, 3]


def test_an_open_input_the_upstream_circuit_does_not_feed_stays_open() -> None:
    downstream = _sink(name="downstream")
    downstream.Q(7) | std.PhaseRotation(pi / 2)

    whole = downstream @ _source()

    assert whole.open_modes == [7]
    assert whole.input_arity == 1


def test_a_shared_mode_number_that_is_not_a_connection_is_refused() -> None:
    """The same number naming a different physical mode in each circuit is a mistake, not a shift."""
    upstream = CircuitRepr("upstream")
    upstream.Q(0) | QuantumState.vacuum()
    _ = upstream.Q(0) | std.MeasureHomodyne(0.0)  # mode 0 is consumed, so it is not an output
    downstream = CircuitRepr("downstream")
    downstream.Q(0) | QuantumState.vacuum()  # mode 0 is closed, so it is not an open input

    with pytest.raises(RuntimeError, match="without being a connection"):
        _ = downstream @ upstream


def test_the_refusal_reports_both_sides() -> None:
    upstream = CircuitRepr("upstream")
    upstream.Q(0) | QuantumState.vacuum()
    downstream = CircuitRepr("downstream")
    downstream.Q(0) | QuantumState.vacuum()

    with pytest.raises(RuntimeError, match=r"Outputs of 'upstream': \[0\]"):
        _ = downstream @ upstream


def test_connecting_nothing_warns_but_composes(caplog: pytest.LogCaptureFixture) -> None:
    upstream = _source(mode=0, name="upstream")
    downstream = _sink(mode=5, name="downstream")

    with caplog.at_level(logging.WARNING, logger="claveles.circuit.program"):
        whole = downstream @ upstream

    assert "connects nothing" in caplog.text
    assert whole.modes == [0, 5]
    assert whole.n_operations == 2


def test_a_connection_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="claveles.circuit.program"):
        _ = _sink() @ _source()

    assert not caplog.text


def test_the_operands_are_left_untouched() -> None:
    upstream = _source(name="upstream")
    downstream = _sink(name="downstream")

    _ = downstream @ upstream

    assert upstream.n_operations == downstream.n_operations == 1
    assert upstream.placements[0].owner is upstream
    assert downstream.placements[0].owner is downstream


def test_feedforward_crossing_from_the_upstream_circuit_is_resolved() -> None:
    upstream = CircuitRepr("upstream")
    upstream.Q(0) | QuantumState.vacuum()
    upstream.Q(1) | QuantumState.vacuum()
    upstream.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=pi)
    measured = upstream.Q(0) | std.MeasureHomodyne(0.0)
    downstream = CircuitRepr("downstream")
    downstream.Q(1) | std.PhaseRotation(double(measured))

    whole = downstream @ upstream

    nlffs = whole.proto().nlffs
    assert [(ff.from_operation, ff.to_operation) for ff in nlffs] == [(1, 2)]


def test_composing_with_something_that_is_not_a_circuit_is_rejected() -> None:
    with pytest.raises(TypeError):
        _ = _sink() @ 1  # ty: ignore[unsupported-operator]


def test_a_circuit_can_be_fed_into_itself() -> None:
    """A circuit that takes a mode open and leaves it carrying a state can run twice in series."""
    circuit = _sink(name="twice")

    whole = circuit @ circuit

    assert whole.modes == [0]
    assert whole.n_operations == 2
    assert whole.input_arity == 1
    assert whole.get_operation(0) is not whole.get_operation(1)


def test_composition_is_associative() -> None:
    first = _source(name="first")
    second = _sink(name="second")
    third = _sink(name="third")

    left_first = third @ (second @ first)
    right_first = (third @ second) @ first

    assert left_first.modes == right_first.modes
    assert [op.name() for op in left_first] == [op.name() for op in right_first]


def test_the_result_serializes() -> None:
    whole = _sink() @ _source()

    proto = whole.proto()

    assert proto.n_modes == 1
    assert len(proto.initial_states) == 1
    assert len(proto.operations) == 1


def test_an_open_mode_of_the_result_still_needs_a_default() -> None:
    downstream = _sink(name="downstream")
    downstream.Q(7) | std.PhaseRotation(pi / 2)

    whole = downstream @ _source()

    with pytest.raises(ValueError, match="no state placed"):
        whole.proto()
    assert whole.proto(default_state=BosonicState.vacuum()).n_modes == 2
