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
#
# Portions of this file are derived from MQC3,
# originally licensed under the MIT License.

"""Test states for modes of quantum circuit."""

import inspect
from math import cos, sin

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from claveles.circuit.ops import intrinsic
from claveles.circuit.program import CircuitRepr
from claveles.circuit.state import (
    BosonicState,
    FockState,
    GaussianState,
    HardwareConstrainedSqueezedState,
    QuantumState,
    construct_quantum_state_from_proto,
)
from claveles.constant import hbar


def test_gaussian_state() -> None:
    mean = np.array([1, 2, 3, 4], dtype=np.complex128)
    cov = np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 10]], dtype=np.float64)
    state = GaussianState(mean=mean, cov=cov)
    assert isinstance(state, GaussianState)
    assert state.n_modes == 2


def test_gaussian_state_error() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        GaussianState(mean=np.array([1, 2, 3], dtype=np.complex128), cov=np.zeros((3, 3), dtype=np.float64))

    with pytest.raises(ValueError):  # noqa: PT011
        GaussianState(mean=np.array([1, 2], dtype=np.complex128), cov=np.zeros((4, 4), dtype=np.float64))


def test_gaussian_state_vacuum() -> None:
    state = GaussianState.vacuum()
    assert isinstance(state, GaussianState)
    assert state.n_modes == 1
    assert np.allclose(state.mean, np.zeros(2, dtype=np.complex128))
    assert np.allclose(state.cov, 0.5 * np.eye(2, dtype=np.float64))


def test_gaussian_state_squeezed() -> None:
    r = np.log(2)
    hbar = 1.0

    # p-squeezed
    phi_90 = np.pi / 2
    state = GaussianState.squeezed(r, phi_90)
    assert isinstance(state, GaussianState)
    assert state.n_modes == 1
    assert np.allclose(state.mean, np.zeros(2, dtype=np.complex128))
    assert np.allclose(state.cov, 0.5 * hbar * np.array([[4, 0], [0, 0.25]], dtype=np.float64))

    # x-squeezed
    phi_0 = 0.0
    state = GaussianState.squeezed(r, phi_0)
    assert np.allclose(state.cov, hbar * 0.5 * np.array([[0.25, 0], [0, 4]], dtype=np.float64))

    # 60 degree squeezing angle
    phi_60 = np.pi / 3
    state = GaussianState.squeezed(r, phi_60)
    rot = np.array([[cos(phi_60), -sin(phi_60)], [sin(phi_60), cos(phi_60)]], dtype=np.float64)
    cov_expected = hbar * 0.5 * rot @ np.array([[0.25, 0], [0, 4]], dtype=np.float64) @ rot.T
    assert np.allclose(state.cov, cov_expected)


def test_gaussian_state_is_vacuum() -> None:
    state = GaussianState.vacuum()
    assert state.is_vacuum()

    mean = np.array([1, 2], dtype=np.complex128)
    cov = np.diag([1, 3]).astype(np.float64)
    non_vacuum = GaussianState(mean, cov)

    assert not non_vacuum.is_vacuum()


def test_bosonic_state() -> None:
    coeffs = np.array([1 / 2] * 2, np.complex128)
    gaussian_states = [
        GaussianState(
            mean=np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128),
            cov=np.array(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                ],
                dtype=np.float64,
            ),
        ),
        GaussianState(
            mean=np.array([2.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128),
            cov=np.array(
                [
                    [2.0, 3.0],
                    [3.0, 5.0],
                ],
                dtype=np.float64,
            ),
        ),
    ]
    state = BosonicState(coeffs, gaussian_states)

    assert isinstance(state, BosonicState)
    assert state.n_modes == 1
    assert state.n_peaks == 2

    for i in range(state.n_peaks):
        assert state.get_coeff(0) == coeffs[0]
        g_state = state.get_gaussian_state(i)
        assert np.allclose(g_state.mean, gaussian_states[i].mean)
        assert np.allclose(g_state.cov, gaussian_states[i].cov)


def test_bosonic_state_vacuum() -> None:
    """Test vacuum of BosonicState."""
    state = BosonicState.vacuum()

    assert isinstance(state, BosonicState)
    assert state.n_modes == 1
    assert state.n_peaks == 1
    assert np.allclose(state.coeffs, np.array([1.0 + 0.0j], dtype=np.complex128))
    assert np.allclose(state.gaussian_states[0].mean, np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128))
    assert np.allclose(state.gaussian_states[0].cov, 0.5 * np.eye(2, dtype=np.float64))


def test_bosonic_state_is_vacuum() -> None:
    """Test is_vacuum of BosonicState."""
    state = BosonicState.vacuum()
    assert state.is_vacuum()

    coeffs = np.array([1 / 2] * 2, np.complex128)
    gaussian_states = [
        GaussianState(
            mean=np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128),
            cov=np.array(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                ],
                dtype=np.float64,
            ),
        ),
        GaussianState(
            mean=np.array([2.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128),
            cov=np.array(
                [
                    [2.0, 3.0],
                    [3.0, 5.0],
                ],
                dtype=np.float64,
            ),
        ),
    ]
    non_vacuum = BosonicState(coeffs, gaussian_states)
    assert not non_vacuum.is_vacuum()


@pytest.mark.parametrize(
    argnames=("x", "p", "parity"), argvalues=[(0, 0, 0.5), (1, 0, 0), (0, 1, 1), (1, 2, 3), (1, 1, 0.75)]
)
def test_bosonic_state_cat(x: float, p: float, parity: float) -> None:
    state = BosonicState.cat(x, p, parity)
    assert state.n_modes == 1
    if np.isclose(x, 0) and np.isclose(p, 0):
        assert state.is_vacuum()
    else:
        assert not state.is_vacuum()
        assert state.n_peaks == 4
        assert np.isclose(np.sum(state.coeffs), 1.0)
        for i in range(state.n_peaks):
            assert np.allclose(state.get_gaussian_state(i).cov, GaussianState.vacuum().cov)
        # coeffs should be integer if parity is an integer.
        if np.isclose(parity, np.round(parity)):
            assert (np.abs(np.imag(state.coeffs)) < 1e-15).all()
        else:
            assert (np.abs(np.imag(state.coeffs)) >= 1e-15).any()


def test_bosonic_state_error() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        BosonicState(
            coeffs=np.array([1 / np.sqrt(3)] * 3, dtype=np.complex128),
            gaussian_states=[GaussianState.vacuum(), GaussianState.vacuum()],
        )

    with pytest.raises(ValueError):  # noqa: PT011
        BosonicState(
            coeffs=np.array([1, 2], dtype=np.complex128),
            gaussian_states=[GaussianState.vacuum(), GaussianState.vacuum()],
        )

    mean = np.array([1, 2, 3, 4], dtype=np.complex128)
    cov = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):  # noqa: PT011
        BosonicState(
            coeffs=np.array([1 / np.sqrt(2)] * 2, dtype=np.complex128),
            gaussian_states=[GaussianState(mean, cov), GaussianState.vacuum()],
        )


def test_placed_states_reports_each_mode() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0) | BosonicState.vacuum()
    circuit.Q(1) | intrinsic.PhaseRotation(0.0)

    expected = BosonicState.vacuum()
    states = circuit.placed_states
    assert sorted(states) == [0, 1]
    actual = states[0]

    # Mode 1 was only acted on, so its state has to come from outside.
    assert states[1] is None
    assert circuit.open_modes == [1]

    assert isinstance(actual, BosonicState)
    assert actual.n_modes == expected.n_modes
    assert np.allclose(actual.coeffs, expected.coeffs)
    assert len(actual.gaussian_states) == 1
    assert np.allclose(actual.gaussian_states[0].mean, expected.gaussian_states[0].mean)
    assert np.allclose(actual.gaussian_states[0].cov, expected.gaussian_states[0].cov)


def test_placed_states_is_empty_for_an_untouched_circuit() -> None:
    circuit = CircuitRepr("test")

    assert circuit.placed_states == {}
    assert circuit.open_modes == []
    assert circuit.input_arity == 0
    assert circuit.output_arity == 0


def test_place_state_on_a_mode() -> None:
    circuit = CircuitRepr("test")

    mean = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    cov = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
        ],
        dtype=np.float64,
    )
    gaussian_state = GaussianState(mean, cov)
    new_state = BosonicState(
        np.array([1], dtype=np.complex128),
        [gaussian_state],
    )
    circuit.Q(0) | new_state
    actual = circuit.placed_states[0]

    assert isinstance(actual, BosonicState)
    assert actual.n_modes == new_state.n_modes
    assert np.allclose(actual.coeffs, new_state.coeffs)
    assert len(actual.gaussian_states) == 1
    assert np.allclose(actual.gaussian_states[0].mean, new_state.gaussian_states[0].mean)
    assert np.allclose(actual.gaussian_states[0].cov, new_state.gaussian_states[0].cov)


def test_place_multi_mode_bosonic_state_is_accepted() -> None:
    circuit = CircuitRepr("test")
    coeffs = np.array([1 / 2] * 2, dtype=np.complex128)
    gaussian_states = [
        GaussianState(
            np.array([1, 0, 0, 1], dtype=np.complex128),
            np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 10]], dtype=np.float64),
        ),
        GaussianState(
            np.array([1, 2, 3, 4], dtype=np.complex128),
            np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 10]], dtype=np.float64),
        ),
    ]
    new_state = BosonicState(coeffs, gaussian_states)

    # A bosonic state may span several modes, which is how a correlation between them is kept.
    circuit.Q(0, 1) | new_state
    modes, component = circuit.placed_components[0]
    assert modes == [0, 1]
    assert component.n_modes == 2


def test_convert_gaussian_state() -> None:
    mean = np.array([1 / np.sqrt(2)] * 2, dtype=np.complex128)
    cov = np.array([[0.5, 0], [0, 0.25]], dtype=np.float64)
    state = GaussianState(mean, cov)

    pb_state = state.proto()
    reconstructed_state = GaussianState.construct_from_proto(pb_state)

    assert isinstance(reconstructed_state, GaussianState)
    assert reconstructed_state.n_modes == state.n_modes
    assert np.allclose(reconstructed_state.mean, state.mean)
    assert np.allclose(reconstructed_state.cov, state.cov)


def test_convert_bosonic_state() -> None:
    mean = np.array([1 / np.sqrt(2)] * 2, dtype=np.complex128)
    cov = np.array([[0.5, 0], [0, 0.25]], dtype=np.float64)
    g_state = GaussianState(mean, cov)

    coeffs = np.array([1], dtype=np.complex128)

    state = BosonicState(coeffs, [g_state])
    pb_state = state.proto()

    reconstructed_state = BosonicState.construct_from_proto(pb_state)

    assert isinstance(reconstructed_state, BosonicState)
    assert reconstructed_state.n_modes == state.n_modes
    assert np.allclose(reconstructed_state.coeffs, state.coeffs)
    assert len(reconstructed_state.gaussian_states) == 1
    assert np.allclose(reconstructed_state.gaussian_states[0].mean, g_state.mean)
    assert np.allclose(reconstructed_state.gaussian_states[0].cov, g_state.cov)


def test_extract_mode_at() -> None:
    mean0 = np.array([1.0 + 2.0j, -3.5 + 0.5j, 2.2 - 1.1j, 0.0 + 4.0j, -1.0 - 2.0j, 3.3 + 0.0j], dtype=np.complex128)
    cov0 = np.array(
        [
            [6.0, 2.0, 1.0, 0.5, 1.5, 0.0],
            [2.0, 5.0, 0.5, 1.0, 0.0, 1.2],
            [1.0, 0.5, 4.0, 0.8, 0.6, 0.3],
            [0.5, 1.0, 0.8, 3.5, 0.9, 0.7],
            [1.5, 0.0, 0.6, 0.9, 4.5, 1.1],
            [0.0, 1.2, 0.3, 0.7, 1.1, 3.8],
        ],
        dtype=np.float64,
    )
    g_state0 = GaussianState(mean0, cov0)

    mean1 = np.array([0.5 - 1.2j, -2.0 + 3.3j, 4.1 + 0.0j, -0.7 - 0.7j, 1.5 + 2.5j, -3.0 + 1.0j], dtype=np.complex128)
    cov1 = np.array(
        [
            [5.0, -1.0, 0.5, 0.0, -0.3, 0.7],
            [-1.0, 4.5, -0.6, 0.9, 0.2, 0.0],
            [0.5, -0.6, 3.8, 0.4, -0.5, 0.3],
            [0.0, 0.9, 0.4, 4.2, 0.6, -0.8],
            [-0.3, 0.2, -0.5, 0.6, 3.9, 0.4],
            [0.7, 0.0, 0.3, -0.8, 0.4, 4.1],
        ],
        dtype=np.float64,
    )
    g_state1 = GaussianState(mean1, cov1)

    coeffs = np.array([0.7, 0.3], dtype=np.complex128)
    state = BosonicState(coeffs, [g_state0, g_state1])
    extracted_state0 = state.extract_mode_at(0)
    extracted_state1 = state.extract_mode_at(1)
    extracted_state2 = state.extract_mode_at(2)

    assert extracted_state0.get_coeff(0) == pytest.approx(0.7)
    assert extracted_state0.get_coeff(1) == pytest.approx(0.3)
    assert extracted_state1.get_coeff(0) == pytest.approx(0.7)
    assert extracted_state1.get_coeff(1) == pytest.approx(0.3)
    assert extracted_state2.get_coeff(0) == pytest.approx(0.7)
    assert extracted_state2.get_coeff(1) == pytest.approx(0.3)
    with pytest.raises(IndexError):
        extracted_state0.get_coeff(2)
    with pytest.raises(IndexError):
        extracted_state1.get_coeff(2)
    with pytest.raises(IndexError):
        extracted_state2.get_coeff(2)

    assert_array_equal(extracted_state0.get_gaussian_state(0).mean, mean0[[0, 3]])
    assert_array_equal(extracted_state1.get_gaussian_state(0).mean, mean0[[1, 4]])
    assert_array_equal(extracted_state2.get_gaussian_state(0).mean, mean0[[2, 5]])
    assert_array_equal(extracted_state0.get_gaussian_state(1).mean, mean1[[0, 3]])
    assert_array_equal(extracted_state1.get_gaussian_state(1).mean, mean1[[1, 4]])
    assert_array_equal(extracted_state2.get_gaussian_state(1).mean, mean1[[2, 5]])
    with pytest.raises(IndexError):
        extracted_state0.get_gaussian_state(2)
    with pytest.raises(IndexError):
        extracted_state1.get_gaussian_state(2)
    with pytest.raises(IndexError):
        extracted_state2.get_gaussian_state(2)

    assert_array_equal(extracted_state0.get_gaussian_state(0).cov, cov0[np.ix_([0, 3], [0, 3])])
    assert_array_equal(extracted_state1.get_gaussian_state(0).cov, cov0[np.ix_([1, 4], [1, 4])])
    assert_array_equal(extracted_state2.get_gaussian_state(0).cov, cov0[np.ix_([2, 5], [2, 5])])
    assert_array_equal(extracted_state0.get_gaussian_state(1).cov, cov1[np.ix_([0, 3], [0, 3])])
    assert_array_equal(extracted_state1.get_gaussian_state(1).cov, cov1[np.ix_([1, 4], [1, 4])])
    assert_array_equal(extracted_state2.get_gaussian_state(1).cov, cov1[np.ix_([2, 5], [2, 5])])


@pytest.mark.parametrize(
    argnames=("theta", "phi"),
    argvalues=[(0, 0), (np.pi, 0), (np.pi / 2, 0), (np.pi / 2, np.pi), (np.pi / 4, np.pi / 3)],
)
def test_bosonic_state_gkp(theta: float, phi: float) -> None:
    state = BosonicState.gkp(theta=theta, phi=phi)
    assert state.n_modes == 1
    assert state.n_peaks > 0
    assert np.isclose(np.sum(state.coeffs), 1.0)
    for i in range(state.n_peaks):
        assert state.get_gaussian_state(i).n_modes == 1


def test_bosonic_state_gkp_epsilon() -> None:
    state_large = BosonicState.gkp(theta=0, phi=0, epsilon=0.3)
    state_small = BosonicState.gkp(theta=0, phi=0, epsilon=0.1)
    assert state_small.n_peaks > state_large.n_peaks


def test_bosonic_state_gkp_symmetry() -> None:
    state_0 = BosonicState.gkp(theta=0, phi=0, epsilon=0.1, ampl_cutoff=1e-4)
    state_1 = BosonicState.gkp(theta=np.pi, phi=0, epsilon=0.1, ampl_cutoff=1e-4)
    assert state_0.n_peaks == state_1.n_peaks


def _two_mode_bosonic() -> BosonicState:
    # Two-mode single-peak state: x-mean of the mode at row 0 is 1.0, at row 1 is 3.0.
    mean = np.array([1.0, 3.0, 0.0, 0.0], dtype=np.complex128)
    cov = np.eye(4, dtype=np.float64)
    return BosonicState(np.array([1.0], dtype=np.complex128), [GaussianState(mean, cov)])


def test_modes_default_to_consecutive_numbering() -> None:
    assert QuantumState([_two_mode_bosonic()]).modes == [0, 1]


def test_modes_may_have_gaps() -> None:
    # Surviving modes 0 and 2, mode 1 having been removed mid-circuit.
    state = QuantumState([_two_mode_bosonic()], [[0, 2]])
    assert state.modes == [0, 2]
    assert state.n_modes == 2


def test_mode_list_length_must_match_the_component() -> None:
    with pytest.raises(ValueError, match="spans 2 mode"):
        QuantumState([_two_mode_bosonic()], [[0, 1, 2]])


def test_modes_must_not_repeat_across_components() -> None:
    single = BosonicState.vacuum()
    with pytest.raises(ValueError, match="must not repeat"):
        QuantumState([single, single], [[1], [1]])


def test_extract_mode_selects_by_the_index_in_the_state() -> None:
    state = QuantumState([_two_mode_bosonic()], [[0, 2]])
    # Mode 0 sits at row 0 (x = 1.0); mode 2 sits at row 1 (x = 3.0).
    extracted_0, _ = state.extract_mode(0).states[0]
    extracted_2, _ = state.extract_mode(2).states[0]
    assert isinstance(extracted_0, BosonicState)
    assert isinstance(extracted_2, BosonicState)
    assert np.isclose(extracted_0.get_gaussian_state(0).mean[0], 1.0)
    assert np.isclose(extracted_2.get_gaussian_state(0).mean[0], 3.0)


def test_extract_mode_rejects_an_absent_index() -> None:
    state = QuantumState([_two_mode_bosonic()], [[0, 2]])
    # Mode 1 was removed; addressing it by its original index must raise instead of silently
    # returning a different surviving mode.
    with pytest.raises(ValueError, match="Mode 1 is not present"):
        state.extract_mode(1)


def test_saved_state_proto_carries_the_modes() -> None:
    pb = _two_mode_bosonic().proto()
    # `modes` is output-only: the simulator emits it, the SDK input `proto()` does not.
    assert list(pb.modes) == []
    # Simulate a result proto that carries the surviving modes.
    pb.modes.extend([0, 2])
    assert construct_quantum_state_from_proto(pb).modes == [0, 2]


def test_saved_state_proto_without_modes_is_consecutive() -> None:
    pb = _two_mode_bosonic().proto()
    assert construct_quantum_state_from_proto(pb).modes == [0, 1]


def _fock_pair() -> tuple[FockState, FockState]:
    return FockState.coherent(0.7 + 0.2j, cutoff_dim=5), FockState.squeezed(0.3, 0.0, cutoff_dim=4)


def test_ket_is_the_outer_product_in_ascending_mode_order() -> None:
    first, second = _fock_pair()
    # The components are listed in the other order, so the axes have to be sorted by mode.
    state = QuantumState([second, first], [[1], [0]])

    ket = state.ket()
    assert ket.shape == (5, 4)
    assert np.allclose(ket, np.multiply.outer(first.state_vector(), second.state_vector()))


def test_ket_closes_up_a_gap_in_the_modes() -> None:
    first, second = _fock_pair()
    state = QuantumState([first, second], [[2], [0]])

    # Mode 0 is the second component, mode 2 the first, and the axes follow that order.
    assert np.allclose(state.ket(), np.multiply.outer(second.state_vector(), first.state_vector()))


def test_dm_is_the_kronecker_product() -> None:
    first, second = _fock_pair()
    state = QuantumState([first, second], [[0], [1]])

    dm = state.dm()
    assert dm.shape == (20, 20)
    assert np.allclose(dm, np.kron(first.density_matrix(), second.density_matrix()))
    assert np.allclose(dm, np.outer(state.ket().ravel(), state.ket().ravel().conjugate()))


def test_dm_trace_is_the_product_of_the_component_weights() -> None:
    first, second = _fock_pair()
    state = QuantumState([first, second], [[0], [1]])

    # Neither representation is normalized, and the weights multiply.
    expected = np.trace(first.density_matrix()) * np.trace(second.density_matrix())
    assert np.trace(state.dm()) == pytest.approx(expected)


def test_dm_accepts_a_mixed_component() -> None:
    first, _ = _fock_pair()
    mixed = FockState.from_density_matrix(np.diag([0.6, 0.4]).astype(np.complex128), cutoff_dim=2)
    state = QuantumState([mixed, first], [[0], [1]])

    assert np.allclose(state.dm(), np.kron(mixed.density_matrix(), first.density_matrix()))


def test_ket_rejects_a_mixed_component() -> None:
    mixed = FockState.from_density_matrix(np.diag([0.6, 0.4]).astype(np.complex128), cutoff_dim=2)

    with pytest.raises(ValueError, match="is mixed"):
        QuantumState([mixed]).ket()


def test_dm_of_a_single_mode_matches_the_partial_trace_of_the_whole() -> None:
    first, second = _fock_pair()
    state = QuantumState([first, second], [[0], [1]])

    joint = state.dm().reshape(5, 4, 5, 4)
    traced = np.einsum("ijkj->ik", joint)
    # extract_mode drops the other mode's weight, which the partial trace keeps.
    assert np.allclose(traced, first.density_matrix() * np.trace(second.density_matrix()))
    assert np.allclose(state.extract_mode(0).dm(), first.density_matrix())


def test_ket_and_dm_reject_a_non_fock_component() -> None:
    first, _ = _fock_pair()
    state = QuantumState([first, BosonicState.vacuum()], [[0], [1]])

    with pytest.raises(ValueError, match="BosonicState"):
        state.ket()
    with pytest.raises(ValueError, match="BosonicState"):
        state.dm()


def test_ket_and_dm_of_a_zero_mode_state_are_trivial() -> None:
    state = QuantumState([], [])

    assert state.ket().shape == ()
    assert state.ket() == pytest.approx(1.0)
    assert np.allclose(state.dm(), np.array([[1.0]], dtype=np.complex128))


def test_is_pure_and_is_gaussian_of_each_representation() -> None:
    first, _ = _fock_pair()
    mixed = FockState.from_density_matrix(np.diag([0.6, 0.4]).astype(np.complex128), cutoff_dim=2)
    thermal = GaussianState(np.zeros(2, dtype=np.complex128), np.eye(2, dtype=np.float64))

    assert QuantumState([first]).is_pure
    assert not QuantumState([first]).is_gaussian
    assert not QuantumState([mixed]).is_pure

    assert QuantumState([GaussianState.squeezed(r=1.2, phi=0.3)]).is_pure
    assert QuantumState([GaussianState.squeezed(r=1.2, phi=0.3)]).is_gaussian
    assert not QuantumState([thermal]).is_pure
    assert QuantumState([thermal]).is_gaussian

    assert QuantumState.vacuum().is_pure
    assert QuantumState.vacuum().is_gaussian
    # A cat state is pure, but several peaks are reported as not pure.
    assert not QuantumState.cat(alpha=2.0).is_pure
    assert not QuantumState.cat(alpha=2.0).is_gaussian

    assert QuantumState([HardwareConstrainedSqueezedState(phi=0.4)]).is_pure
    assert QuantumState([HardwareConstrainedSqueezedState(phi=0.4)]).is_gaussian


def test_is_pure_and_is_gaussian_are_the_conjunction_over_the_components() -> None:
    first, _ = _fock_pair()

    assert QuantumState([], []).is_pure
    assert QuantumState([], []).is_gaussian
    # The Fock component is pure but not Gaussian, so only is_gaussian falls.
    assert (QuantumState.vacuum() & QuantumState([first])).is_pure
    assert not (QuantumState.vacuum() & QuantumState([first])).is_gaussian
    assert not (QuantumState.vacuum() & QuantumState.cat(alpha=2.0)).is_pure


def test_multi_mode_gaussian_purity_uses_the_determinant() -> None:
    # Two-mode squeezed vacuum is pure; adding noise to one quadrature breaks it.
    assert QuantumState.epr(r=1.0, repr="gaussian").is_pure

    epr = QuantumState.epr(r=1.0, repr="gaussian").state_list[0]
    assert isinstance(epr, GaussianState)
    noisy = GaussianState(epr.mean, epr.cov + np.diag([0.3, 0.0, 0.0, 0.0]))
    assert not noisy.is_pure


def _three_component_state() -> QuantumState:
    return QuantumState(
        [BosonicState.vacuum(), BosonicState.vacuum(), BosonicState.vacuum()],
        [[0], [4], [3]],
        {4: "epr1"},
    )


def test_idx_shift_moves_every_mode_and_its_name() -> None:
    shifted = _three_component_state().idx_shift(2)

    assert shifted.mode_idx_list == [[2], [6], [5]]
    assert shifted.mode_names == {6: "epr1"}


def test_idx_shift_leaves_the_original_alone_and_round_trips() -> None:
    state = _three_component_state()
    there_and_back = state.idx_shift(3).idx_shift(-3)

    assert state.mode_idx_list == [[0], [4], [3]]
    assert there_and_back.mode_idx_list == state.mode_idx_list
    assert there_and_back.mode_names == state.mode_names


def test_idx_shift_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError, match="below zero"):
        _three_component_state().idx_shift(-1)


def test_idx_align_closes_up_the_gaps() -> None:
    aligned = _three_component_state().idx_align()

    # Modes 0, 3, 4 become 0, 1, 2, each keeping its place in the ascending order.
    assert aligned.mode_idx_list == [[0], [2], [1]]
    assert aligned.mode_names == {2: "epr1"}
    assert aligned.idx_align().mode_idx_list == aligned.mode_idx_list


def test_idx_align_keeps_the_order_within_a_multi_mode_component() -> None:
    state = QuantumState([_two_mode_bosonic(), BosonicState.vacuum()], [[2, 6], [4]])

    assert state.idx_align().mode_idx_list == [[0, 2], [1]]


def test_wigner_func_of_one_mode_matches_the_meshed_version() -> None:
    from claveles.circuit.visualize_state import meshed_wigner  # noqa: PLC0415

    xvec = np.linspace(-4, 4, 21)
    pvec = np.linspace(-3, 3, 17)
    x_grid, p_grid = np.meshgrid(xvec, pvec)

    for basic in (BosonicState.cat(x=1.0, p=0.5, parity=0), FockState.fock(2, cutoff_dim=8)):
        wigner = QuantumState([basic]).wigner_func()(x_grid, p_grid)
        assert wigner.shape == (len(pvec), len(xvec))
        assert np.allclose(wigner, meshed_wigner(basic, xvec, pvec))


def test_wigner_func_of_a_gaussian_component_matches_a_single_peak_bosonic_one() -> None:
    from claveles.circuit.visualize_state import meshed_wigner  # noqa: PLC0415

    xvec = np.linspace(-4, 4, 21)
    pvec = np.linspace(-3, 3, 17)
    x_grid, p_grid = np.meshgrid(xvec, pvec)
    gaussian = GaussianState.squeezed(r=0.4, phi=0.7)
    single_peak = BosonicState(np.fromiter([1], dtype=np.complex128), [gaussian])

    wigner = QuantumState([gaussian]).wigner_func()(x_grid, p_grid)
    assert np.allclose(wigner, meshed_wigner(single_peak, xvec, pvec))


def test_wigner_func_of_a_two_mode_state_is_normalized() -> None:
    state = QuantumState([FockState.fock(1, cutoff_dim=6), BosonicState.vacuum()], [[1], [0]])
    grid = np.linspace(-5, 5, 61)
    axes = np.meshgrid(grid, grid, grid, grid, indexing="ij", sparse=True)

    values = state.wigner_func()(*axes)
    step = grid[1] - grid[0]
    assert values.sum() * step**4 == pytest.approx(1.0, abs=1e-3)


def test_wigner_func_of_a_two_mode_gaussian_component_is_normalized() -> None:
    grid = np.linspace(-5, 5, 61)
    axes = np.meshgrid(grid, grid, grid, grid, indexing="ij", sparse=True)

    # A single component spanning both modes, which the Gaussian mixture handles in 2n variables.
    values = QuantumState.epr(r=0.3).wigner_func()(*axes)
    step = grid[1] - grid[0]
    assert values.sum() * step**4 == pytest.approx(1.0, abs=1e-3)


def test_wigner_func_takes_the_quadratures_in_ascending_mode_order() -> None:
    displaced = QuantumState.coherent(alpha=1.5)
    state = QuantumState.concatenate([QuantumState.vacuum(), displaced], mode_idx_list=[[3], [1]])

    wigner = state.wigner_func()
    # Mode 1 is the displaced one and comes first, so its x sits in the first argument.
    peak = wigner(np.sqrt(2) * 1.5, 0.0, 0.0, 0.0)
    swapped = wigner(0.0, np.sqrt(2) * 1.5, 0.0, 0.0)
    assert peak > swapped


def test_wigner_func_rejects_the_wrong_number_of_variables() -> None:
    wigner = QuantumState.vacuum().wigner_func()

    with pytest.raises(ValueError, match="quadrature values"):
        wigner(0.0)


def test_wigner_func_rejects_a_state_it_cannot_expand() -> None:
    with pytest.raises(ValueError, match="HardwareConstrainedSqueezedState"):
        QuantumState([HardwareConstrainedSqueezedState(phi=0.0)]).wigner_func()

    two_mode_fock = FockState(cutoff_dims=[3, 3], state_vector=np.zeros(9, dtype=np.complex128))
    with pytest.raises(ValueError, match="2-mode Fock state"):
        QuantumState([two_mode_fock]).wigner_func()


def test_draw_wigner_matches_the_figure_helper() -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from claveles.circuit.visualize_state import make_wigner_figure, meshed_wigner  # noqa: PLC0415

    xvec = np.linspace(-5, 5, 40)
    pvec = np.linspace(-5, 5, 40)
    basic = BosonicState.cat(x=1.0, p=0.0, parity=0)

    figure = QuantumState([basic]).draw_wigner(xvec, pvec, title="cat")
    expected = make_wigner_figure(basic, xvec, pvec, title="cat")
    assert figure.axes[0].get_title() == expected.axes[0].get_title() == "cat"
    assert figure.axes[0].get_xlabel() == expected.axes[0].get_xlabel()
    # The plotted values are the ones the helper computes.
    assert np.allclose(QuantumState([basic]).wigner_func()(*np.meshgrid(xvec, pvec)), meshed_wigner(basic, xvec, pvec))
    plt.close(figure)
    plt.close(expected)


def test_draw_wigner_defaults_to_its_own_grid() -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure = QuantumState.vacuum().draw_wigner()
    assert figure.axes[0].get_xlabel() == "x"
    plt.close(figure)


def test_draw_wigner_rejects_more_than_one_mode() -> None:
    state = QuantumState.vacuum() & QuantumState.vacuum()

    with pytest.raises(ValueError, match="Only single mode states"):
        state.draw_wigner()


def test_draw_wigner_rejects_a_hardware_constrained_state() -> None:
    with pytest.raises(ValueError, match="HardwareConstrainedSqueezedState"):
        QuantumState([HardwareConstrainedSqueezedState(phi=0.0)]).draw_wigner()


def test_bosonic_purity_is_not_the_conjunction_over_the_peaks() -> None:
    mixture = BosonicState(
        np.array([0.5, 0.5], dtype=np.complex128),
        [GaussianState.coherent(1.5), GaussianState.coherent(-1.5)],
    )

    # Each peak has the vacuum covariance and so is pure on its own, but the mixture is not.
    assert all(peak.is_pure for peak in mixture.gaussian_states)
    assert not mixture.is_pure


def test_a_peak_may_sit_below_the_vacuum() -> None:
    gkp = BosonicState.gkp(epsilon=0.1)
    peak = gkp.get_gaussian_state(0)

    # A peak carries whatever covariance the decomposition needs, so its purity says nothing about
    # the state it is part of: this one is narrower than the vacuum, which no state is.
    assert np.linalg.det(peak.cov) < (0.5 * hbar) ** 2
    assert not peak.is_pure
    assert not gkp.is_pure


def test_cutoff_dims_are_reported_in_ascending_mode_order() -> None:
    pair = QuantumState.fock(n=1, cutoff_dim=3) & QuantumState.coherent(0.5, repr="fock", cutoff_dim=4)

    assert pair.cutoff_dims == [3, 4]
    assert pair.ket().shape == tuple(pair.cutoff_dims)


def test_cutoff_dims_follow_the_modes_not_the_component_order() -> None:
    # The component holds mode 2 first, so its own first cutoff belongs to the higher mode.
    state = QuantumState(
        [FockState(cutoff_dims=[3, 4], state_vector=np.ones(12, dtype=np.complex128))],
        [[2, 0]],
    )

    assert state.cutoff_dims == [4, 3]
    assert state.ket().shape == tuple(state.cutoff_dims)


def test_cutoff_dims_report_none_for_an_unset_conversion_cutoff() -> None:
    state = QuantumState.fock(n=1, cutoff_dim=3) & QuantumState.vacuum()

    assert state.cutoff_dims == [3, None]


def test_cutoff_dims_report_none_for_a_state_that_has_no_cutoff() -> None:
    # The machine's squeezed state names a level rather than amplitudes, so no cutoff applies to it.
    assert QuantumState.hardware_squeezed().cutoff_dims == [None]


def test_cutoff_dim_matches_the_component_for_a_single_mode() -> None:
    state = QuantumState.coherent(0.5, repr="fock", cutoff_dim=7)

    component = state.state_list[0]
    assert isinstance(component, FockState)
    assert state.cutoff_dim == 7
    assert state.cutoff_dim == component.cutoff_dim


def test_cutoff_dim_rejects_more_than_one_mode() -> None:
    pair = QuantumState.fock(n=1, cutoff_dim=3) & QuantumState.fock(n=1, cutoff_dim=4)

    with pytest.raises(ValueError, match="holds 2 modes"):
        _ = pair.cutoff_dim


def test_mean_puts_every_x_before_every_p() -> None:
    state = QuantumState.squeezed(r=0.5) & QuantumState.coherent(1.0)

    # Mode 1 is the coherent one, so its x mean lands at index 1 rather than index 2.
    assert state.mean.real == pytest.approx([0.0, np.sqrt(2 * hbar), 0.0, 0.0])
    assert state.mean.dtype == np.complex128
    assert state.cov.dtype == np.float64


@pytest.mark.parametrize("mode_idx_list", [[[0, 2], [1]], [[2, 0], [1]]])
def test_mean_and_cov_interleave_the_components_by_mode(mode_idx_list: list[list[int]]) -> None:
    state = QuantumState.concatenate(
        [QuantumState.epr(r=0.7, phi=0.3), QuantumState.coherent(1 + 0.5j)],
        mode_idx_list=mode_idx_list,
    )

    # A component orders its own x's then its own p's, while the state orders every mode's x then
    # every mode's p, so rebuilding from the assembled pair has to reproduce the original exactly.
    rebuilt = QuantumState.gaussian(state.mean, state.cov)
    quadratures = np.random.default_rng(0).normal(size=(8, 6))
    for point in quadratures:
        assert rebuilt.wigner_func()(*point) == pytest.approx(state.wigner_func()(*point))


def test_cov_has_no_correlation_between_two_components() -> None:
    state = QuantumState.squeezed(r=0.5) & QuantumState.coherent(1.0)

    # The components are separable, so only the blocks within one of them carry anything.
    assert_array_equal(state.cov[np.ix_([0, 2], [1, 3])], np.zeros((2, 2)))
    assert_array_equal(state.cov[np.ix_([1, 3], [0, 2])], np.zeros((2, 2)))


def test_mean_and_cov_of_a_zero_mode_state_are_empty() -> None:
    state = QuantumState()

    assert state.mean.shape == (0,)
    assert state.cov.shape == (0, 0)


def test_mean_and_cov_reject_a_multi_peak_component() -> None:
    cat = QuantumState.cat(alpha=2.0)

    with pytest.raises(ValueError, match=r"mode\(s\) \[0\] is a BosonicState with 4 peaks"):
        _ = cat.mean
    with pytest.raises(ValueError, match="every component has to be Gaussian"):
        _ = cat.cov


def test_mean_and_cov_reject_a_fock_component() -> None:
    state = QuantumState.fock(n=1, cutoff_dim=4)

    with pytest.raises(ValueError, match="held in the Fock basis"):
        _ = state.mean


def test_mean_and_cov_reject_a_hardware_constrained_component_that_calls_itself_gaussian() -> None:
    state = QuantumState.hardware_squeezed()

    # The trap: the state answers is_gaussian with True while having no mean to report, so is_gaussian
    # is not the test for whether these can be read.
    assert state.is_gaussian
    with pytest.raises(ValueError, match="squeezing level the machine decides"):
        _ = state.mean
    with pytest.raises(ValueError, match="squeezing level the machine decides"):
        _ = state.cov


def test_from_state_vector_inverts_ket() -> None:
    pair = QuantumState.fock(n=1, cutoff_dim=3) & QuantumState.coherent(0.5, repr="fock", cutoff_dim=4)

    rebuilt = QuantumState.from_state_vector(pair.ket(), pair.cutoff_dims)

    assert rebuilt.cutoff_dims == pair.cutoff_dims
    assert rebuilt.ket() == pytest.approx(pair.ket())
    assert rebuilt.dm() == pytest.approx(pair.dm())


def test_from_density_matrix_inverts_dm() -> None:
    pair = QuantumState.fock(n=1, cutoff_dim=3) & QuantumState.coherent(0.5, repr="fock", cutoff_dim=4)

    rebuilt = QuantumState.from_density_matrix(pair.dm(), pair.cutoff_dims)

    assert rebuilt.dm() == pytest.approx(pair.dm())
    # Purity follows how the state is stored, so a density matrix is mixed even when it is pure.
    assert not rebuilt.is_pure


def test_from_state_vector_of_one_mode_takes_a_plain_cutoff() -> None:
    amplitudes = np.zeros(6, dtype=np.complex128)
    amplitudes[0] = amplitudes[3] = 1 / np.sqrt(2)

    state = QuantumState.from_state_vector(amplitudes, cutoff_dim=6)

    assert state.n_modes == 1
    assert state.is_pure
    assert state.ket() == pytest.approx(amplitudes)


def test_from_state_vector_spans_one_mode_per_cutoff() -> None:
    joint = np.zeros((2, 3), dtype=np.complex128)
    joint[0, 0] = joint[1, 2] = 1 / np.sqrt(2)

    state = QuantumState.from_state_vector(joint, cutoff_dim=[2, 3])

    # One component covering both modes, which is what keeps them correlated.
    assert state.modes == [0, 1]
    assert len(state.state_list) == 1
    assert state.cutoff_dims == [2, 3]


def test_hardware_squeezed_holds_the_state_the_machine_supplies() -> None:
    state = QuantumState.hardware_squeezed(phi=0.25)

    assert state.n_modes == 1
    component = state.state_list[0]
    assert isinstance(component, HardwareConstrainedSqueezedState)
    assert component.phi == pytest.approx(0.25)


def test_hardware_squeezed_can_be_placed_and_serialized() -> None:
    circuit = CircuitRepr("hardware")
    circuit.Q(0) | QuantumState.hardware_squeezed(phi=0.25)
    circuit.Q(0) | intrinsic.PhaseRotation(0.1)

    proto = circuit.proto()

    assert proto.initial_states[0].squeezed.theta == pytest.approx(np.pi / 2 - 0.25)


def test_hardware_squeezed_takes_neither_a_repr_nor_a_cutoff() -> None:
    parameters = inspect.signature(QuantumState.hardware_squeezed).parameters

    # It carries no amplitudes and no covariance, so there is nothing to hold otherwise or to truncate.
    assert list(parameters) == ["phi"]
