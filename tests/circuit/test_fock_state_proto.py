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

"""Round-trips of a Fock state through the initial-state protobuf form."""

from __future__ import annotations

import numpy as np
import pytest

from claveles.circuit.state import (
    FockState,
    construct_initial_state_from_proto,
    construct_proto_from_initial_state,
)


def _round_trip(state: FockState) -> FockState:
    """Serialize through InitialState and back.

    The reader also reports the modes the entry named; a single-mode state leaves them implicit, so
    what comes back here is None alongside the state.
    """
    modes, result = construct_initial_state_from_proto(construct_proto_from_initial_state(state))
    assert modes is None
    assert isinstance(result, FockState)
    return result


def test_number_state_round_trip() -> None:
    """A number state stays a number, rather than arriving as an expanded basis vector."""
    restored = _round_trip(FockState.fock(n=3, cutoff_dim=5))

    assert restored.is_pure
    assert restored.cutoff_dim == 5
    assert restored.number == 3


def test_state_vector_survives_round_trip() -> None:
    """A pure superposition must not decay into a number state on the way through proto."""
    sv = np.array([0.0, 1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    restored = _round_trip(FockState.from_state_vector(sv, cutoff_dim=3))

    assert restored.is_pure
    assert restored.number is None
    np.testing.assert_allclose(restored.state_vector(), sv, atol=1e-6)


def test_mixed_state_round_trip() -> None:
    dm = np.diag([0.25, 0.75, 0.0]).astype(np.complex128)
    restored = _round_trip(FockState.from_density_matrix(dm, cutoff_dim=3))

    assert not restored.is_pure
    np.testing.assert_allclose(restored.density_matrix(), dm, atol=1e-6)


def test_mixed_state_with_coherences_round_trip() -> None:
    sv = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    dm = np.outer(sv, sv.conjugate()).astype(np.complex128)
    restored = _round_trip(FockState.from_density_matrix(dm, cutoff_dim=2))

    np.testing.assert_allclose(restored.density_matrix(), dm, atol=1e-6)


def test_vacuum_round_trip_is_a_number_state() -> None:
    restored = _round_trip(FockState.vacuum(cutoff_dim=4))

    assert restored.number == 0
    assert restored.cutoff_dim == 4


def test_from_density_matrix_normalize() -> None:
    dm = np.diag([1.0, 3.0]).astype(np.complex128)
    state = FockState.from_density_matrix(dm, cutoff_dim=2, normalize=True)

    assert np.isclose(np.trace(state.density_matrix()), 1.0)
    assert np.isclose(state.density_matrix()[1, 1], 0.75)


def test_from_density_matrix_normalize_rejects_zero_trace() -> None:
    dm = np.zeros((2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="trace is zero"):
        FockState.from_density_matrix(dm, cutoff_dim=2, normalize=True)


def test_repr_distinguishes_representations() -> None:
    number = FockState.fock(n=1, cutoff_dim=2)
    pure = FockState.from_state_vector(np.array([1.0, 0.0], dtype=np.complex128), cutoff_dim=2)
    mixed = FockState.from_density_matrix(np.eye(2, dtype=np.complex128) / 2.0, cutoff_dim=2)

    assert "number=1" in repr(number)
    assert "state_vector" in repr(pure)
    assert "density_matrix" in repr(mixed)
