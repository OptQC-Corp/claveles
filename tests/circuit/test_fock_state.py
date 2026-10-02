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

"""Tests for the FockState basic state class."""

from __future__ import annotations

import numpy as np
import pytest

from claveles.circuit.state import BasicState, FockState, QuantumState
from claveles.circuit.visualize_state import wigner_values


def test_pure_fock_state() -> None:
    sv = np.array([0.0, 1.0, 0.0], dtype=np.complex128)
    state = FockState(cutoff_dims=[3], state_vector=sv)

    assert isinstance(state, BasicState)
    assert state.is_pure
    assert state.n_modes == 1
    assert state.cutoff_dim == 3
    assert state.cutoff_dims == [3]
    np.testing.assert_array_equal(state.state_vector(), sv)


def test_pure_fock_state_density_matrix_is_outer_product() -> None:
    sv = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    state = FockState(cutoff_dims=[2], state_vector=sv)

    rho = state.density_matrix()
    assert rho.shape == (2, 2)
    np.testing.assert_allclose(rho, np.outer(sv, sv.conjugate()))
    assert np.isclose(np.trace(rho), 1.0)


def test_sub_unit_weight_is_preserved_in_both_representations() -> None:
    """A truncated state's weight is reported rather than normalized away.

    The norm of a pure state and the trace of a mixed one both carry it, and for a pure state the
    trace of the density matrix equals the squared norm of the state vector.
    """
    weight = 0.25
    sv = np.array([0.5, 0.0], dtype=np.complex128)  # squared norm 0.25
    pure = FockState(cutoff_dims=[2], state_vector=sv)
    assert np.isclose(np.linalg.norm(pure.state_vector()) ** 2, weight)
    assert np.isclose(np.trace(pure.density_matrix()), weight)

    mixed = FockState(cutoff_dims=[2], density_matrix=np.diag([0.15, 0.1]).astype(np.complex128))
    assert np.isclose(np.trace(mixed.density_matrix()), weight)


def test_mixed_fock_state() -> None:
    dm = np.diag([0.5, 0.5, 0.0]).astype(np.complex128)
    state = FockState(cutoff_dims=[3], density_matrix=dm)

    assert not state.is_pure
    np.testing.assert_array_equal(state.density_matrix(), dm)


def test_mixed_fock_state_has_no_state_vector() -> None:
    dm = np.eye(2, dtype=np.complex128) / 2.0
    state = FockState(cutoff_dims=[2], density_matrix=dm)

    with pytest.raises(ValueError, match="mixed"):
        state.state_vector()


def test_number_state_keeps_the_number_and_expands_on_demand() -> None:
    """A number state stays a number, because that is the form the simulator accepts."""
    state = FockState.fock(n=2, cutoff_dim=4)

    assert state.number == 2
    assert state.is_pure
    assert state.n_modes == 1
    np.testing.assert_array_equal(state.state_vector(), np.array([0, 0, 1, 0], dtype=np.complex128))
    assert state.proto().number == 2


def test_number_state_is_rejected_outside_the_cutoff() -> None:
    with pytest.raises(ValueError, match="must be less than cutoff_dim"):
        FockState.fock(n=4, cutoff_dim=4)


def test_number_state_is_single_mode_only() -> None:
    with pytest.raises(ValueError, match="single mode"):
        FockState(cutoff_dims=[2, 3], number=1)


def test_requires_exactly_one_representation() -> None:
    sv = np.array([1.0, 0.0], dtype=np.complex128)
    dm = np.eye(2, dtype=np.complex128) / 2.0

    with pytest.raises(ValueError, match="Exactly one"):
        FockState(cutoff_dims=[2])
    with pytest.raises(ValueError, match="Exactly one"):
        FockState(cutoff_dims=[2], state_vector=sv, density_matrix=dm)
    with pytest.raises(ValueError, match="Exactly one"):
        FockState(cutoff_dims=[2], state_vector=sv, number=0)


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="state_vector shape"):
        FockState(cutoff_dims=[3], state_vector=np.array([[1.0, 0.0]], dtype=np.complex128))
    with pytest.raises(ValueError, match="density_matrix shape"):
        FockState(cutoff_dims=[3], density_matrix=np.eye(2, dtype=np.complex128))


def test_rejects_empty_cutoff_dims() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        FockState(cutoff_dims=[], state_vector=np.array([], dtype=np.complex128))


def test_multi_mode_pure_state_is_reshaped_per_mode() -> None:
    # |0,1> + |1,0> over cutoffs (2, 3), normalized.
    dims = (2, 3)
    sv = np.zeros(dims, dtype=np.complex128)
    sv[0, 1] = 1.0 / np.sqrt(2.0)
    sv[1, 0] = 1.0 / np.sqrt(2.0)

    state = FockState(cutoff_dims=[2, 3], state_vector=sv)

    assert state.n_modes == 2
    assert state.cutoff_dims == [2, 3]
    assert state.state_vector().shape == dims
    # Mode 0 is the most significant axis.
    assert state.state_vector()[0, 1] == pytest.approx(1.0 / np.sqrt(2.0))


def test_multi_mode_flat_state_vector_is_accepted() -> None:
    dims = (2, 3)
    flat = np.zeros(6, dtype=np.complex128)
    flat[1] = 1.0  # (n0, n1) = (0, 1) with mode 0 most significant

    state = FockState(cutoff_dims=[2, 3], state_vector=flat)

    assert state.state_vector().shape == dims
    assert state.state_vector()[0, 1] == pytest.approx(1.0)


def test_multi_mode_density_matrix_is_square_over_joint_basis() -> None:
    dims = (2, 3)
    sv = np.zeros(dims, dtype=np.complex128)
    sv[0, 1] = 1.0 / np.sqrt(2.0)
    sv[1, 0] = 1.0 / np.sqrt(2.0)
    state = FockState(cutoff_dims=[2, 3], state_vector=sv)

    rho = state.density_matrix()
    assert rho.shape == (6, 6)
    assert np.isclose(np.trace(rho), 1.0)
    # A pure state's density matrix is idempotent.
    np.testing.assert_allclose(rho @ rho, rho, atol=1e-12)


def test_a_multi_mode_state_uses_the_multi_mode_message() -> None:
    """There are two messages, and a joint state goes out as the one that can hold it."""
    sv = np.zeros(6, dtype=np.complex128)
    sv[0] = 1.0
    state = FockState(cutoff_dims=[2, 3], state_vector=sv)

    # proto() builds the single-mode message, which has nowhere to put a second mode.
    with pytest.raises(ValueError, match="carries a single mode"):
        state.proto()

    joint = state.multi_mode_proto()
    assert list(joint.cutoff_dims) == [2, 3]
    assert joint.HasField("state_vector")
    assert len(joint.state_vector.amplitudes) == 6


def test_cutoff_dim_rejects_multi_mode() -> None:
    sv = np.zeros(6, dtype=np.complex128)
    sv[0] = 1.0
    state = FockState(cutoff_dims=[2, 3], state_vector=sv)

    with pytest.raises(ValueError, match="cutoff_dims"):
        _ = state.cutoff_dim


def test_cutoff_dims_is_a_copy() -> None:
    state = FockState(cutoff_dims=[2], state_vector=np.array([1.0, 0.0], dtype=np.complex128))

    state.cutoff_dims.append(9)

    assert state.cutoff_dims == [2]


def test_modes_belong_to_the_surrounding_state() -> None:
    """A basic state does not know where it sits; the containing QuantumState assigns that."""
    state = FockState.fock(n=1, cutoff_dim=3)

    assert not hasattr(state, "modes")
    assert QuantumState([state], [[2]]).modes == [2]


def test_from_state_vector_takes_a_cutoff_per_mode() -> None:
    joint = np.arange(6, dtype=np.complex128).reshape(2, 3)

    state = FockState.from_state_vector(joint, cutoff_dim=[2, 3])

    assert state.n_modes == 2
    assert state.cutoff_dims == [2, 3]
    np.testing.assert_array_equal(state.state_vector(), joint)


def test_from_state_vector_accepts_a_shaped_or_a_flat_joint_array() -> None:
    joint = np.arange(6, dtype=np.complex128).reshape(2, 3)

    shaped = FockState.from_state_vector(joint, cutoff_dim=[2, 3])
    flat = FockState.from_state_vector(joint.ravel(), cutoff_dim=[2, 3])

    np.testing.assert_array_equal(shaped.state_vector(), flat.state_vector())


def test_from_density_matrix_takes_a_cutoff_per_mode() -> None:
    dm = np.eye(6, dtype=np.complex128) / 6

    state = FockState.from_density_matrix(dm, cutoff_dim=[2, 3])

    assert state.cutoff_dims == [2, 3]
    assert not state.is_pure
    np.testing.assert_array_equal(state.density_matrix(), dm)


def test_from_state_vector_accepts_an_integral_that_is_not_an_int() -> None:
    # A dimension read out of a numpy array is integral without being an `int`, and iterating one
    # raises, so the two forms cannot be told apart with `isinstance(..., int)`.
    cutoff = np.zeros((6,)).shape[0]

    # The annotation names `int`, while any integral works: that is what this pins down.
    state = FockState.from_state_vector(
        np.zeros(6, dtype=np.complex128),
        np.int64(cutoff),  # ty: ignore[invalid-argument-type]
    )

    assert state.cutoff_dims == [6]


def test_from_state_vector_rejects_a_cutoff_that_names_no_dimensions() -> None:
    with pytest.raises(TypeError, match="whole number, or a sequence of them"):
        FockState.from_state_vector(np.zeros(6, dtype=np.complex128), 6.5)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match="whole number, or a sequence of them"):
        FockState.from_state_vector(np.zeros(6, dtype=np.complex128), "6")  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="names no modes"):
        FockState.from_state_vector(np.zeros(6, dtype=np.complex128), [])


def _reduced_by_brute_force(dm: np.ndarray, cutoff_dims: list[int], position: int) -> np.ndarray:
    """Sum the joint density matrix over every basis state of the other modes, one term at a time."""
    joint = dm.reshape(tuple(cutoff_dims) * 2)
    others = [range(d) for k, d in enumerate(cutoff_dims) if k != position]
    dim = cutoff_dims[position]
    reduced = np.zeros((dim, dim), dtype=np.complex128)
    for rest in np.ndindex(*(len(o) for o in others)):
        for i in range(dim):
            for j in range(dim):
                row = list(rest)
                col = list(rest)
                row.insert(position, i)
                col.insert(position, j)
                reduced[i, j] += joint[tuple(row + col)]
    return reduced


def test_extract_mode_at_traces_out_the_other_mode_of_an_entangled_state() -> None:
    # (|0,2> - |2,0>) / sqrt(2): either mode alone is an equal mixture of |0> and |2>.
    sv = np.zeros((3, 3), dtype=np.complex128)
    sv[0, 2] = 1 / np.sqrt(2)
    sv[2, 0] = -1 / np.sqrt(2)
    state = FockState(cutoff_dims=[3, 3], state_vector=sv)

    for position in (0, 1):
        reduced = state.extract_mode_at(position)
        assert reduced.cutoff_dims == [3]
        assert not reduced.is_pure
        np.testing.assert_allclose(reduced.density_matrix(), np.diag([0.5, 0.0, 0.5]), atol=1e-12)


def test_extract_mode_at_of_a_product_keeps_the_factor_and_the_weight_of_the_rest() -> None:
    first = np.array([0.6, 0.8j], dtype=np.complex128)
    second = np.array([0.5, 0.0, 0.5j], dtype=np.complex128)  # squared norm 0.5
    state = FockState(cutoff_dims=[2, 3], state_vector=np.multiply.outer(first, second))

    np.testing.assert_allclose(state.extract_mode_at(0).density_matrix(), np.outer(first, first.conj()) * 0.5)
    np.testing.assert_allclose(state.extract_mode_at(1).density_matrix(), np.outer(second, second.conj()))


@pytest.mark.parametrize("position", [0, 1, 2])
def test_extract_mode_at_matches_a_brute_force_partial_trace(position: int) -> None:
    rng = np.random.default_rng(217)
    cutoff_dims = [2, 3, 4]
    sv = rng.normal(size=cutoff_dims) + 1j * rng.normal(size=cutoff_dims)
    pure = FockState(cutoff_dims=cutoff_dims, state_vector=sv)
    reference = _reduced_by_brute_force(pure.density_matrix(), cutoff_dims, position)

    np.testing.assert_allclose(pure.extract_mode_at(position).density_matrix(), reference)
    # The same state held as a density matrix takes the other branch and must agree with it.
    mixed_form = FockState(cutoff_dims=cutoff_dims, density_matrix=pure.density_matrix())
    np.testing.assert_allclose(mixed_form.extract_mode_at(position).density_matrix(), reference)


def test_extract_mode_at_of_a_mixed_state_keeps_its_trace() -> None:
    rng = np.random.default_rng(218)
    cutoff_dims = [3, 2]
    m = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    dm = m @ m.conj().T
    dm *= 0.7 / np.trace(dm).real
    state = FockState(cutoff_dims=cutoff_dims, density_matrix=dm)

    for position in (0, 1):
        reduced = state.extract_mode_at(position).density_matrix()
        np.testing.assert_allclose(reduced, _reduced_by_brute_force(dm, cutoff_dims, position))
        assert np.trace(reduced) == pytest.approx(0.7)


def test_extract_mode_at_returns_a_single_mode_state_as_it_is() -> None:
    state = FockState.fock(2, cutoff_dim=4)

    assert state.extract_mode_at(0) is state


@pytest.mark.parametrize("position", [-1, 2])
def test_extract_mode_at_refuses_a_position_outside_the_state(position: int) -> None:
    state = FockState(cutoff_dims=[2, 2], state_vector=np.eye(2, dtype=np.complex128) / np.sqrt(2))

    with pytest.raises(ValueError, match="out of range"):
        state.extract_mode_at(position)


def test_container_extract_mode_narrows_a_multi_mode_fock_component() -> None:
    sv = np.zeros((3, 3), dtype=np.complex128)
    sv[0, 2] = sv[2, 0] = 1 / np.sqrt(2)
    state = QuantumState([FockState(cutoff_dims=[3, 3], state_vector=sv)], [[3, 7]], {7: "idler"})

    extracted = state.extract_mode(7)
    assert extracted.modes == [7]
    assert extracted.mode_names == {7: "idler"}
    np.testing.assert_allclose(extracted.dm(), np.diag([0.5, 0.0, 0.5]), atol=1e-12)
    # A single mode, so it can now be drawn.
    assert np.isfinite(extracted.wigner_func()(0.0, 0.0))


def test_the_wigner_function_of_a_multi_mode_fock_state_points_at_extracting_a_mode() -> None:
    sv = np.zeros((3, 3), dtype=np.complex128)
    sv[0, 2] = sv[2, 0] = 1 / np.sqrt(2)
    basic = FockState(cutoff_dims=[3, 3], state_vector=sv)
    state = QuantumState([basic], [[0, 1]])

    with pytest.raises(ValueError, match=r"QuantumState\.extract_mode\(\)"):
        state.wigner_func()
    with pytest.raises(ValueError, match=r"FockState\.extract_mode_at\(\)"):
        wigner_values(basic, 0.0, 0.0, 0.0, 0.0)
    # Following either hint gives the same one-mode Wigner function.
    assert state.extract_mode(0).wigner_func()(0.3, -0.2) == pytest.approx(
        wigner_values(basic.extract_mode_at(0), 0.3, -0.2)
    )
