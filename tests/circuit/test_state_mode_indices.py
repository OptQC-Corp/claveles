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

"""Tests for the mode indices a combined state ends up on, and for who may call the joining methods.

Laying states side by side keeps the first state's indices and lifts each of the others clear of them,
the rule :meth:`CircuitRepr.__and__` follows. Renumbering from 0 instead would move a state whose
indices mean something, such as the one a simulation hands back.
"""

from __future__ import annotations

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import QuantumState
from claveles.client import SimulatorClient


def _gapped() -> QuantumState:
    """Return a two-mode state sitting on modes 0 and 2, so that mode 1 is a gap."""
    return QuantumState.concatenate(
        [QuantumState.squeezed(r=0.5), QuantumState.vacuum()],
        mode_idx_list=[[0], [2]],
    )


def test_two_states_built_on_their_own_are_laid_out_from_zero() -> None:
    # Both sit on mode 0, which is the case `&` exists for.
    combined = QuantumState.squeezed(r=0.5) & QuantumState.vacuum()

    assert combined.modes == [0, 1]


def test_the_first_state_keeps_the_indices_it_has() -> None:
    combined = _gapped() & QuantumState.vacuum()

    assert combined.modes == [0, 2, 3]


def test_a_state_laid_after_a_gap_starts_above_it_rather_than_filling_it() -> None:
    combined = _gapped() & _gapped()

    # The second copy is lifted by 3, so its own gap is carried along with it.
    assert combined.modes == [0, 2, 3, 5]


def test_laying_out_three_states_lifts_each_above_the_ones_before_it() -> None:
    combined = QuantumState.tensor(QuantumState.vacuum(), _gapped(), QuantumState.vacuum())

    assert combined.modes == [0, 1, 3, 4]


def test_an_empty_state_takes_up_no_modes() -> None:
    empty = QuantumState([])

    assert (empty & _gapped()).modes == [0, 2]
    assert (_gapped() & empty).modes == [0, 2]
    assert (_gapped() & empty & QuantumState.vacuum()).modes == [0, 2, 3]


def test_aliases_follow_the_modes_they_name() -> None:
    left = QuantumState.concatenate(
        [QuantumState.squeezed(r=0.5), QuantumState.vacuum()],
        mode_idx_list=[[0], [2]],
        mode_names={2: "ancilla"},
    )
    right = QuantumState.concatenate([QuantumState.vacuum()], mode_names={0: "probe"})

    combined = left & right

    assert combined.modes == [0, 2, 3]
    assert combined.mode_names == {2: "ancilla", 3: "probe"}


def test_a_state_a_simulation_reported_keeps_its_modes_when_laid_beside_another() -> None:
    """The motivating case: a measured-away mode leaves a gap that must not be closed silently."""
    # The simulator wheel is an optional extra, so CI runs the suite without it.
    pytest.importorskip("claveles_simulator")

    circuit = CircuitRepr("readout")
    for mode in range(3):
        circuit.Q(mode) | QuantumState.squeezed(r=0.5)
    circuit.Q(1) | std.MeasureHomodyne(0.0)

    saved = SimulatorClient(n_shots=1, state_save_policy="all").run(circuit).shot_results[0].state
    assert saved is not None
    assert saved.modes == [0, 2]

    # Mode 2 stays mode 2 rather than being pulled down into the gap mode 1 left.
    assert (saved & QuantumState.vacuum()).modes == [0, 2, 3]


def test_idx_align_is_how_the_gaps_are_closed() -> None:
    combined = _gapped() & QuantumState.vacuum()

    assert combined.idx_align().modes == [0, 1, 2]


def test_indexed_keeps_every_state_where_it_is() -> None:
    combined = QuantumState.tensor(_gapped(), QuantumState.vacuum().idx_shift(5), indexed=True)

    assert combined.modes == [0, 2, 5]


def test_indexed_refuses_states_whose_modes_overlap() -> None:
    with pytest.raises(ValueError, match="must not repeat"):
        QuantumState.tensor(QuantumState.vacuum(), QuantumState.vacuum(), indexed=True)


def test_mode_idx_list_cannot_be_combined_with_laying_out() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        QuantumState.concatenate([QuantumState.vacuum()], mode_idx_list=[[3]], indexed=False)


def test_tensor_means_the_same_thing_through_the_class_and_through_an_instance() -> None:
    left, right = _gapped(), QuantumState.vacuum()

    assert QuantumState.tensor(left, right).modes == left.tensor(right).modes == (left & right).modes


def test_concatenate_refuses_to_be_reached_through_an_instance() -> None:
    """Reaching it that way would drop the instance and quietly return a state of the wrong width."""
    left, right = QuantumState.squeezed(r=0.5), QuantumState.vacuum()

    with pytest.raises(TypeError, match="through an instance"):
        left.concatenate([right])

    # The class-level call is the one that works, and is unchanged.
    assert QuantumState.concatenate([left, right], indexed=False).modes == [0, 1]
