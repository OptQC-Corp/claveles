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

"""Tests for FockState normalize option."""

from __future__ import annotations

import numpy as np
import pytest

from claveles.circuit.state import FockState


def test_coherent_default_analytically_normalized() -> None:
    """Coherent state default has sub-unit norm due to truncation (analytically normalized)."""
    state = FockState.coherent(alpha=2.0, cutoff_dim=10)
    norm = np.linalg.norm(state.state_vector())
    assert norm < 1.0


def test_coherent_normalize_true() -> None:
    """Coherent state with normalize=True has exact unit norm."""
    state = FockState.coherent(alpha=2.0, cutoff_dim=10, normalize=True)
    assert np.isclose(np.linalg.norm(state.state_vector()), 1.0)


def test_squeezed_default_analytically_normalized() -> None:
    """Squeezed state default has sub-unit norm due to truncation."""
    state = FockState.squeezed(r=1.5, phi=0.0, cutoff_dim=10)
    norm = np.linalg.norm(state.state_vector())
    assert norm < 1.0


def test_squeezed_normalize_true() -> None:
    """Squeezed state with normalize=True has exact unit norm."""
    state = FockState.squeezed(r=1.5, phi=0.0, cutoff_dim=10, normalize=True)
    assert np.isclose(np.linalg.norm(state.state_vector()), 1.0)


def test_cat_default_analytically_normalized() -> None:
    """Cat state default applies analytical factor (norm < 1 at low cutoff)."""
    state = FockState.cat(alpha=2.0, parity=0.0, cutoff_dim=30)
    assert np.isclose(np.linalg.norm(state.state_vector()), 1.0, atol=1e-6)


def test_cat_default_low_cutoff_norm_below_one() -> None:
    """Cat state default at low cutoff has norm < 1 (analytical factor doesn't compensate truncation)."""
    state = FockState.cat(alpha=2.0, parity=0.0, cutoff_dim=5)
    norm = np.linalg.norm(state.state_vector())
    assert norm < 1.0


def test_cat_normalize_true() -> None:
    """Cat state with normalize=True has exact unit norm regardless of cutoff."""
    state = FockState.cat(alpha=2.0, parity=0.0, cutoff_dim=5, normalize=True)
    assert np.isclose(np.linalg.norm(state.state_vector()), 1.0)


def test_cat_matches_sf() -> None:
    """Cat state default matches Strawberry Fields construction."""
    from scipy.special import factorial  # noqa: PLC0415

    alpha = 2.0
    parity = 0.0
    cutoff = 5

    state = FockState.cat(alpha=alpha, parity=parity, cutoff_dim=cutoff)

    # Expected values transcribed from strawberryfields.utils.states.cat_state
    # (Apache License 2.0, Copyright 2019 Xanadu Quantum Technologies Inc.). Written out here rather
    # than imported, since strawberryfields is not a dependency of this package.
    ns = np.arange(cutoff)
    temp = np.exp(-0.5 * abs(alpha) ** 2)
    n_sf = temp / np.sqrt(2 * (1 + np.cos(np.pi * parity) * temp**4))
    c1 = alpha**ns / np.sqrt(factorial(ns))
    c2 = (-alpha) ** ns / np.sqrt(factorial(ns))
    sf_sv = (c1 + np.exp(1j * np.pi * parity) * c2) * n_sf

    np.testing.assert_allclose(state.state_vector(), sf_sv, atol=1e-14)


def test_gkp_always_normalized() -> None:
    """GKP state is always numerically normalized (no analytical factor exists)."""
    state = FockState.gkp(epsilon=0.3, cutoff_dim=20)
    assert np.isclose(np.linalg.norm(state.state_vector()), 1.0)


def test_from_state_vector_default_preserves() -> None:
    """from_state_vector default preserves the input vector."""
    sv = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
    state = FockState.from_state_vector(sv, cutoff_dim=3)
    np.testing.assert_array_equal(state.state_vector(), sv)


def test_from_state_vector_normalize_true() -> None:
    """from_state_vector with normalize=True produces unit norm."""
    sv = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
    state = FockState.from_state_vector(sv, cutoff_dim=3, normalize=True)
    assert np.isclose(np.linalg.norm(state.state_vector()), 1.0)


def test_from_state_vector_normalize_zero_vector() -> None:
    """from_state_vector with normalize=True on zero vector returns zero."""
    sv = np.zeros(5, dtype=np.complex128)
    state = FockState.from_state_vector(sv, cutoff_dim=5, normalize=True)
    np.testing.assert_array_equal(state.state_vector(), sv)


def test_from_state_vector_normalize_uses_the_norm_of_the_whole_joint_array() -> None:
    """A joint state is scaled as one state, not mode by mode."""
    joint = np.ones((2, 3), dtype=np.complex128)

    state = FockState.from_state_vector(joint, cutoff_dim=[2, 3], normalize=True)

    assert np.isclose(np.linalg.norm(state.state_vector()), 1.0)


def test_from_state_vector_normalize_leaves_a_joint_zero_array_alone() -> None:
    """An all-zero joint array has no norm to scale by, so it is passed through."""
    joint = np.zeros((2, 3), dtype=np.complex128)

    state = FockState.from_state_vector(joint, cutoff_dim=[2, 3], normalize=True)

    np.testing.assert_array_equal(state.state_vector(), joint)


def test_from_density_matrix_normalize_uses_the_trace_of_the_joint_matrix() -> None:
    """The trace of the whole matrix is what is scaled, not a per-mode one."""
    dm = np.eye(6, dtype=np.complex128)

    state = FockState.from_density_matrix(dm, cutoff_dim=[2, 3], normalize=True)

    assert np.isclose(np.trace(state.density_matrix()).real, 1.0)


def test_from_density_matrix_normalize_rejects_a_joint_matrix_of_zero_trace() -> None:
    """A zero trace cannot be scaled to one, whatever the number of modes."""
    with pytest.raises(ValueError, match="trace is zero"):
        FockState.from_density_matrix(np.zeros((6, 6), dtype=np.complex128), cutoff_dim=[2, 3], normalize=True)
