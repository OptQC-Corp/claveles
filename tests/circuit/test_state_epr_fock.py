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

"""Tests for the EPR state in the Fock representation.

The three representations have to describe one state, so the Fock amplitudes are checked against the
covariance the Gaussian representation reports rather than against a formula written twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import BosonicState, FockState, GaussianState, QuantumState
from claveles.client import SimulatorClient

_CUTOFF = 24
"""High enough that the truncated tail does not show at the tolerance below."""


def _cutoff_dims(state: QuantumState) -> list[list[int] | None]:
    """Return the conversion cutoff each component records, in component order.

    `cutoff_dims` belongs to the representations that can be converted to the Fock basis rather than
    to `BasicState` itself, so the component has to be narrowed to one of them first.
    """
    dims: list[list[int] | None] = []
    for component, _ in state.states:
        assert isinstance(component, (BosonicState, FockState, GaussianState))
        dims.append(component.cutoff_dims)
    return dims


def _covariance(ket: np.ndarray, cutoff: int) -> np.ndarray:
    """Return the covariance of a two-mode pure state, ordered as (x1, x2, p1, p2).

    Args:
        ket: Joint amplitudes, of shape ``(cutoff, cutoff)``.
        cutoff: Fock dimension of each mode.

    Returns:
        np.ndarray: The 4x4 covariance matrix.
    """
    annihilate = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    create = annihilate.conj().T
    x = (annihilate + create) / np.sqrt(2)
    p = (annihilate - create) / (1j * np.sqrt(2))
    eye = np.eye(cutoff)
    operators = [np.kron(x, eye), np.kron(eye, x), np.kron(p, eye), np.kron(eye, p)]

    psi = ket.reshape(-1)
    psi /= np.linalg.norm(psi)
    means = [np.real(psi.conj() @ (operator @ psi)) for operator in operators]
    return np.array([
        [np.real(psi.conj() @ (((a @ b + b @ a) / 2) @ psi)) - means[i] * means[j] for j, b in enumerate(operators)]
        for i, a in enumerate(operators)
    ])


@pytest.mark.parametrize(("r", "phi"), [(0.0, 0.0), (0.4, 0.0), (0.3, 0.9), (0.7, np.pi / 2)])
def test_the_fock_epr_state_is_the_one_the_other_representations_describe(r: float, phi: float) -> None:
    fock = QuantumState.epr(r=r, phi=phi, repr="fock", cutoff_dim=_CUTOFF)
    gaussian = QuantumState.epr(r=r, phi=phi, repr="gaussian")

    covariance = _covariance(fock.ket(), _CUTOFF)

    component = gaussian.state_list[0]
    assert isinstance(component, GaussianState)
    assert np.allclose(covariance, component.cov.real, atol=1e-9)


def test_it_spans_two_modes() -> None:
    state = QuantumState.epr(r=0.4, repr="fock", cutoff_dim=6)

    assert state.modes == [0, 1]
    assert state.n_modes == 2
    assert _cutoff_dims(state) == [[6, 6]]


def test_only_the_correlated_terms_are_populated() -> None:
    # An EPR state is a superposition of |n, n|: everything off the diagonal is zero.
    ket = QuantumState.epr(r=0.5, repr="fock", cutoff_dim=5).ket()

    assert np.allclose(ket, np.diag(np.diagonal(ket)))


def test_zero_squeezing_is_two_vacuum_modes() -> None:
    ket = QuantumState.epr(r=0.0, repr="fock", cutoff_dim=4).ket()

    expected = np.zeros((4, 4), dtype=np.complex128)
    expected[0, 0] = 1.0
    assert np.allclose(ket, expected)


def test_the_truncated_tail_leaves_the_norm_just_below_one() -> None:
    ket = FockState.epr(0.4, 0.0, 6).state_vector()

    norm = float(np.vdot(ket, ket).real)
    assert norm < 1.0
    assert norm == pytest.approx(1.0, abs=1e-4)


def test_normalize_rescales_the_truncated_state() -> None:
    ket = FockState.epr(0.4, 0.0, 6, normalize=True).state_vector()

    assert float(np.vdot(ket, ket).real) == pytest.approx(1.0)


def test_the_fock_representation_needs_a_cutoff() -> None:
    with pytest.raises(ValueError, match="cutoff_dim"):
        QuantumState.epr(r=0.4, repr="fock")


@pytest.mark.parametrize("repr_", ["bosonic", "gaussian"])
def test_the_other_representations_keep_the_cutoff_for_a_later_conversion(repr_: str) -> None:
    state = QuantumState.epr(r=0.4, repr=repr_, cutoff_dim=8)  # ty: ignore[invalid-argument-type]

    assert _cutoff_dims(state) == [[8, 8]]


def test_a_fock_epr_state_can_be_placed_and_simulated() -> None:
    # The point of having it: a non-Gaussian gate the bosonic representation cannot apply.
    pytest.importorskip("claveles_simulator")

    circuit = CircuitRepr("epr_kerr")
    circuit.Q(0, 1) | QuantumState.epr(r=0.4, repr="fock", cutoff_dim=6)
    circuit.Q(0) | std.Kerr(0.1)

    saved = SimulatorClient(n_shots=1, state_save_policy="all").run(circuit).shot_results[0].state

    assert saved is not None
    assert saved.modes == [0, 1]
    assert _cutoff_dims(saved) == [[6, 6]]
